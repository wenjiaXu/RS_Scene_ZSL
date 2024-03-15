from __future__ import print_function
import os
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import visual_utils
import sys
import random
from visual_utils import ImageFilelist, compute_per_class_acc, compute_per_class_acc_gzsl, \
    prepare_attri_label, add_glasso, add_dim_glasso
# from model_proto import resnet_proto_IoU
from model import CLIP_proto
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data
import json
from main_utils import test_zsl, test_zsl_proposal, \
    calibrated_stacking, test_gzsl, test_gzsl_proposal, \
    calculate_average_IoU, test_with_IoU, set_randomseed, get_loader, \
    get_middle_graph, Loss_fn, Result
from opt import get_opt
import time


opt = get_opt()
# set random seed设置随机种子并初始化，确保在需要时可以重现
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

def main():
    # load data加载数据集
    data = visual_utils.DATA_LOADER(opt)
    opt.test_seen_label = data.test_seen_label  # weird
    opt.test_unseen_label = data.test_unseen_label  # weird

    # define test_classes
    if opt.image_type == 'test_unseen_small_loc':
        test_loc = data.test_unseen_small_loc
        test_classes = data.unseenclasses
    elif opt.image_type == 'test_unseen_loc':
        test_loc = data.test_unseen_loc
        test_classes = data.unseenclasses
    elif opt.image_type == 'test_seen_loc':
        test_loc = data.test_seen_loc
        test_classes = data.seenclasses
    else:
        try:
            sys.exit(0)
        except:
            print("choose the image_type in ImageFileList")

    # prepare the attribute labels
    class_attribute = data.attribute
    attribute_zsl = prepare_attri_label(class_attribute, data.unseenclasses)
    attribute_seen = prepare_attri_label(class_attribute, data.seenclasses)
    attribute_gzsl = torch.transpose(class_attribute, 1, 0)

    # Dataloader for train, test, visual
    trainloader, testloader_unseen, testloader_seen, visloader = get_loader(opt, data)

    # define attribute groups定义属性类：功能、材质、表面
    if opt.dataset == 'RSSDIVCS':
        parts = ['functions', 'materials', 'surface_properties']
        group_dic = json.load(open(opt.image_root+'files/attri_group_3.json'))
        if opt.pro_type == 1:
            group_dic_proposal = {'functions': group_dic['functions']}
        elif opt.pro_type == 2:
            group_dic_proposal = {'materials': group_dic['materials']}
        elif opt.pro_type == 3:
            group_dic_proposal = {'surface_properties': group_dic['surface_properties']}
        elif opt.pro_type == 4:
            group_dic_proposal = {'functions': group_dic['functions'],
                              'materials': group_dic['materials'],
                              'surface_properties': group_dic['surface_properties'],}

    # initialize model初始化模型CLIP
    print('Create Model...')
    model = CLIP_proto(opt, group_dic_proposal)
    criterion = nn.CrossEntropyLoss()
    criterion_regre = nn.MSELoss()

    # optimzation weight, only ['final'] + model.extract are used.
    reg_weight = {'final': {'xe': opt.xe, 'attri': opt.attri, 'regular': opt.regular},
                  'proposal': {'xe': 0,},
                  'layer4': {'l_xe': opt.l_xe, 'attri': opt.l_attri, 'regular': opt.l_regular,
                             'cpt': opt.cpt},  # l denotes layer
                  }
    reg_lambdas = {}
    for name in ['final'] + model.extract:
        reg_lambdas[name] = reg_weight[name]
    # print('reg_lambdas:', reg_lambdas)

    if torch.cuda.is_available():
        model.cuda()
        attribute_zsl = attribute_zsl.cuda()
        attribute_seen = attribute_seen.cuda()
        attribute_gzsl = attribute_gzsl.cuda()

    layer_name = model.extract[0]  # only use one layer currently
    middle_graph = get_middle_graph(reg_weight[layer_name]['cpt'], model)

    # train and test
    result_zsl = Result()
    result_gzsl = Result()


    print('Train and test...')
    for epoch in range(opt.nepoch):
        print('\n', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "Training epoch {}".format(epoch))
        model.train()
        current_lr = opt.classifier_lr * (0.9 ** (epoch // 10))
        model.cur_epoch = epoch

        realtrain = epoch > opt.pretrain_epoch
        if epoch <= opt.pretrain_epoch:   # pretrain ALE for the first several epoches
            optimizer = optim.Adam(params=[model.prototype_vectors[layer_name],
                                           model.ALE_vector[1].weight,
                                           model.ALE_vector[1].bias],
                                   lr=opt.pretrain_lr, betas=(opt.beta1, 0.999))
        else:
            optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=current_lr, betas=(opt.beta1, 0.999))


        # loss for print
        loss_log = {'ave_loss': 0.0, 'l_xe_final': 0.0, 'l_attri_final': 0.0, 'l_regular_final': 0.0,
                    'l_xe_layer': 0.0, 'l_attri_layer': 0.0, 'l_regular_layer': 0.0, 'l_cpt': 0.0,
                    'l_xe_pro': 0.0}

        batch = len(trainloader)
        for i, (batch_input, batch_target, impath) in enumerate(trainloader):
            # map target labels
            optimizer.zero_grad()
            batch_target = visual_utils.map_label(batch_target, data.seenclasses)
            input_v = Variable(batch_input)
            label_v = Variable(batch_target)
            if opt.cuda:
                input_v = input_v.cuda() # image in cpu for clip
                label_v = label_v.cuda()
            # model.zero_grad()

            output, pre_attri, attention, pre_class = model(input_v, attribute_seen)
            label_a = attribute_seen[:, label_v].t()

            loss = Loss_fn(opt, loss_log, reg_weight, criterion, criterion_regre, model,
                           output, pre_attri, attention, pre_class, label_a, label_v,
                           realtrain, middle_graph, parts, group_dic, use_ViT=True)
            loss_log['ave_loss'] += loss.item()

            loss.backward()
            optimizer.step()

            if (i + 1) == batch:
            ###### test #######
            # print("testing")
                model.eval()
                # test zsl
                if not opt.gzsl:
                    acc_ZSL = test_zsl_proposal(opt, model, testloader_unseen, attribute_zsl,
                                                data.unseenclasses, use_proposal=opt.use_proposal)
                    result_zsl.update(epoch, acc_ZSL)
                    print('[Epoch {} Batch {}] ZSL test accuracy is {:.1f}%, Best_acc [{:.1f}% | Epoch-{}]'.
                          format(epoch, i + 1, acc_ZSL, result_zsl.best_acc, result_zsl.best_iter))
                else:
                    # test gzsl
                    print('seen acc:')
                    acc_GZSL_seen = test_gzsl_proposal(opt, model, testloader_seen, attribute_gzsl,
                                                       data.seenclasses, use_proposal=opt.use_proposal)
                    print('unseen acc:')

                    acc_GZSL_unseen = test_gzsl_proposal(opt, model, testloader_unseen, attribute_gzsl,
                                                         data.unseenclasses, use_proposal=opt.use_proposal)

                    if (acc_GZSL_unseen + acc_GZSL_seen) == 0:
                        acc_GZSL_H = 0
                    else:
                        acc_GZSL_H = 2 * acc_GZSL_unseen * acc_GZSL_seen / (
                                acc_GZSL_unseen + acc_GZSL_seen)
                    result_gzsl.update_gzsl(epoch+1, acc_GZSL_unseen, acc_GZSL_seen, acc_GZSL_H)

                    print('[Epoch {} Batch {}] GZSL test accuracy is Unseen: {:.1f} Seen: {:.1f} H:{:.1f}'
                          '\n           Best_H [Unseen: {:.1f}% Seen: {:.1f}% H: {:.1f}% | Epoch-{}]'.
                          format(epoch, i + 1, acc_GZSL_unseen, acc_GZSL_seen, acc_GZSL_H, result_gzsl.best_acc_U, result_gzsl.best_acc_S,
                    result_gzsl.best_acc, result_gzsl.best_iter))

                print('\n[Epoch %d, Batch %5d] Train loss: %.3f '
                          % (epoch, batch, loss_log['ave_loss'] / batch))
                model.train()

if __name__ == '__main__':
    main()

