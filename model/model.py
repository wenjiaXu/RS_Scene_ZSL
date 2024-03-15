import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
import os
import copy
from skimage import measure
import numpy as np
from torchvision.utils import save_image
from torchvision import transforms
import random

import timm

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

def denormalize(x, mean=IMG_MEAN, std=IMG_STD):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class LINEAR_SOFTMAX_ALE(nn.Module):
    def __init__(self, input_dim, attri_dim):
        super(LINEAR_SOFTMAX_ALE, self).__init__()
        self.fc = nn.Linear(input_dim, attri_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, attribute):
        middle = self.fc(x)
        output = self.softmax(middle.mm(attribute))
        return output


class LINEAR_SOFTMAX(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LINEAR_SOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x


class LAYER_ALE(nn.Module):
    def __init__(self, input_dim, attri_dim):
        super(LAYER_ALE, self).__init__()
        self.fc = nn.Linear(input_dim, attri_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, attribute):
        batch_size = x.size(0)
        x = torch.mean(x, dim=1)
        x = x.view(batch_size, -1)
        middle = self.fc(x)
        output = self.softmax(middle.mm(attribute))
        return output


class CLIP_proto(nn.Module):
    def __init__(self, opt, group_dic):
        super(CLIP_proto, self).__init__()
        from transformers import CLIPModel
        assert opt.version in [1, 2], 'Please let opt.version be in [1, 2]'
        if opt.version == 1:
            print('version=1')
            # clip_model = CLIPModel.from_pretrained("flax-community/clip-rsicd")
            clip_model = CLIPModel.from_pretrained("/home/yzj/data/code/ISPRS/ISPRS_2023_clean/flax-community/clip-rsicd")
        elif opt.version == 2:
            print('version=2')
            # clip_model = CLIPModel.from_pretrained("flax-community/clip-rsicd-v2")
            clip_model = CLIPModel.from_pretrained("/home/yzj/data/code/ISPRS/ISPRS_2023_clean/flax-community/clip-rsicd-v2")
        # self.embedding_module = clip_model.vision_model.embeddings
        self.embedding_module = clip_model.vision_model

        self.group = [v for k, v in group_dic.items()]  # list of group id
        self.group_dim = len(self.group)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 02 - load cls weights
        # we left the entry for several layers, but here we only use layer4
        self.dim_dict = {'layer1': 56*56, 'layer2': 28*28, 'layer3': 14*14, 'layer4': 7*7, 'avg_pool': 1*1}
        self.channel_dict = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048, 'avg_pool': 2048}
        self.kernel_size = {'layer1': 56, 'layer2': 28, 'layer3': 14, 'layer4': 7, 'avg_pool': 1}
        self.extract = ['layer4']  # 'layer1', 'layer2', 'layer3', 'layer4'
        self.epsilon = 1e-4

        self.softmax = nn.Softmax(dim=1)
        self.softmax2d = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()
        # only support RSSDIVCS currently
        if opt.dataset == 'RSSDIVCS':
            pool = 'mean'
            dim = 768
            num_classes = opt.attri_dim
            assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
            self.pool = pool
            self.to_latent = nn.Identity()
            self.prototype_vectors = dict()
            for name in self.extract:
                prototype_shape = [num_classes, dim, 1]
                self.prototype_vectors[name] = nn.Parameter(2e-4 * torch.rand(prototype_shape), requires_grad=True)
            self.prototype_vectors = nn.ParameterDict(self.prototype_vectors)
            # prototype_shape = [num_classes, dim, 1]
            # self.prototype_vector_layer4 = nn.Parameter(2e-4 * torch.rand(prototype_shape), requires_grad=True)
            self.ALE_vector = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes)
            )
        self.avg_pool = opt.avg_pool
        weights_init(self.prototype_vectors)
        weights_init(self.ALE_vector)
        # print("self.ALE_vector:\n", self.ALE_vector)

        self.cur_epoch = 0
        self.opt = opt
        self.proposal_start_epoch = opt.pro_start_epoch if opt.pro_start_epoch else 300

    def forward(self, img, attribute):
        """out: predict class, predict attributes, maps, out_feature"""
        last_hidden_state, pooled_output = self.embedding_module(img, return_dict=False)  # [24, 50, 768]
        x = last_hidden_state
        record_features_layer4 = x.transpose(1, 2)  # [24, 1024, 50]
        out = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # [24, 1, 1024]
        out = self.to_latent(out)
        pre_attri_final = self.ALE_vector(out)

        att = F.conv1d(input=record_features_layer4, weight=self.prototype_vectors['layer4'])  # [24, 98, patches]
        pre_attri_layer4 = torch.max(att, dim=2)[0]
        batch, attri_dim, patches = att.size()
        att = att[:, :, 1:]
        attention_layer4 = att.view(batch, attri_dim, 7, 7)
        pre_class_layer4 = self.softmax(pre_attri_layer4.mm(attribute))

        if self.cur_epoch > self.proposal_start_epoch:
            # 需要引入group信息，并且寻找到每组对应的proposal，然后计算各proposal的分类情况
            # print('here1')
            # window_imgs, maps = self.get_proposal(img.cpu(), attention_layer4.cpu(), pre_attri_layer4.cpu())
            # window_imgs = window_imgs.to(self.device)
            with torch.no_grad():
                window_imgs, maps = self.get_proposal(img.clone(), attention_layer4.clone(), pre_attri_layer4.clone())
            # print('here2')
            # print('here3')

            proposal_logits = self.rawcls_net(window_imgs, attribute)
            pre_class_proposal = self.softmax(proposal_logits.mm(attribute))
            output_final = self.softmax(pre_attri_final.mm(attribute) +
                                        self.opt.pro_weight * proposal_logits.mm(attribute))
            return output_final, {'layer4': pre_attri_layer4, 'final': pre_attri_final}, \
                   {'layer4': attention_layer4}, \
                   {'layer4': pre_class_layer4, 'proposal': pre_class_proposal}
        else:
            output_final = self.softmax(pre_attri_final.mm(attribute))
            return output_final, {'layer4': pre_attri_layer4, 'final': pre_attri_final}, \
                   {'layer4': attention_layer4}, \
                   {'layer4': pre_class_layer4}


    def get_proposal(self, img, attention_maps, pre_attri):
        def box(attention_map, maps=[]):
            tmp_map = attention_map
            for i in range(tmp_map.size(0)):
                tmp_map[i] -= torch.min(tmp_map[i])
                tmp_map[i] /= torch.max(tmp_map[i])
            tmp_map = torch.mean(tmp_map, dim=0, keepdim=False)
            maps.append(tmp_map)
            a = torch.mean(tmp_map, dim=(0, 1))
            M = (tmp_map > a * self.opt.pro_thr).cpu().numpy()
            component_labels = measure.label(M)
            properties = measure.regionprops(component_labels)
            if len(properties) == 0:
                bbox = [0, 0, 224, 224]
                return bbox
            areas = []
            for prop in properties:
                areas.append(prop.area)
            max_idx = areas.index(max(areas))
            prop = measure.regionprops((component_labels == (max_idx+1)).astype(int))
            if len(prop) == 0:
                bbox = [0, 0, 224, 224]
            else:
                bbox = prop[0].bbox
            return bbox

        attention_maps = F.interpolate(attention_maps, (224, 224), mode='bilinear', align_corners=True)
        # [24, 312, 224, 224]
        batch_size = img.size(0)
        # print('1')
        window_imgs = torch.zeros([batch_size, 3, 224, 224]).to(self.device)
        # print('2')

        indices = []
        for j in range(self.group_dim):
            # print('pre_attri:', pre_attri.size())
            ids = torch.max(pre_attri[:, self.group[j]], dim=1)[1]  # (batch_size,)
            indices.append(ids)
        # print('3')

        maps = []
        for i in range(batch_size):
            # print('In the image {}'.format(i))
            ids = [self.group[j][indices[j][i]] for j in range(self.group_dim)]
                # print(len(self.group[j]))
            attention_map = attention_maps[i, ids, :]
            # print('5')
            x0, y0, x1, y1 = box(attention_map, maps)
            window_imgs[i:i + 1] = F.interpolate(img[i:i+1, :, x0: x1, y0:y1], size=(224, 224),
                                                    mode='bilinear', align_corners=True)  # [N, 4, 3, 224, 224]
        # print('7')
        return window_imgs, maps


    def rawcls_net(self, img, attribute):
        # 为proposal设计的cls函数
        batch_size = img.size(0)
        last_hidden_state, pooled_output = self.embedding_module(img, return_dict=False)
        x = last_hidden_state
        out = torch.max(x, dim=1)[0]
        out = self.to_latent(out)
        pre_attri = self.ALE_vector(out)
        return pre_attri



    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

