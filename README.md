# **RS_Scene_ZSL**
PyTorch code for Deep Semantic-Visual Alignment for zero-shot remote sensing image scene classification.

**Title:** " Deep Semantic-Visual Alignment for zero-shot remote sensing image scene classification"

**Authors:** Wenjia Xu, Jiuniu Wang, Zhiwei Wei, Mugen Peng, Yirong Wu 

**Abstract:**
Deep neural networks have achieved promising progress in remote sensing (RS) image classification, for which the training process requires abundant samples for each class. However, it is time-consuming and unrealistic to annotate labels for each RS category, given the fact that the RS target database is increasing dynamically. Zero-shot learning (ZSL) allows for identifying novel classes that are not seen during training, which provides a promising solution for the aforementioned problem. However, previous ZSL models mainly depend on manually-labeled attributes or word embeddings extracted from language models to transfer knowledge from seen classes to novel classes. Those class embeddings may not be visually detectable and the annotation process is time-consuming and labor-intensive. Besides, pioneer ZSL models use convolutional neural networks pre-trained on ImageNet, which focus on the main objects appearing in each image, neglecting the background context that also matters in RS scene classification. To address the above problems, we propose to collect visually detectable attributes automatically. We predict attributes for each class by depicting the semantic-visual similarity between attributes and images. In this way, the attribute annotation process is accomplished by machine instead of human as in other methods. Moreover, we propose a Deep SemanticVisual Alignment (DSVA) that take advantage of the self-attention mechanism in the transformer to associate local image regions together, integrating the background context information for prediction. The DSVA model further utilizes the attribute attention maps to focus on the informative image regions that are essential for knowledge transfer in ZSL, and maps the visual images into attribute space to perform ZSL classification. With extensive experiments, we show that our model outperforms other state-of-the-art models by a large margin on a challenging large-scale RS scene classification benchmark. Moreover, we qualitatively verify that the attributes annotated by our network are both class discriminative and semantic related, which benefits the zero-shot knowledge transfer.
## Requirements
Python 3.9

PyTorch = 2.1.0

All experiments are performed with NVIDIA GeForce RTX 4090.
Create environment with requirements.txt, you can follow this [link](https://blog.csdn.net/ft_sunshine/article/details/92215164).

## Prerequisites
- Dataset: please download the dataset [RSSDIVCS](https://ieeexplore.ieee.org/document/9321719), and change the opt.image_root to the dataset root path on your machine
  
- image_embedding and class_embedding: change the  image_embedding_root and image_embedding_root in script to the true path on your machine

- Pre-trained models: please download the [pre-trained models](https://github.com/arampacha/CLIP-rsicd)（CLIP model fine-tuned on the RSICD dataset）and change the model root to the clip root path on your machine

## Code Structures
There are three parts in the code.
 - `model`: It contains the main files of the DSVA network 
 - `script`: It contains the training scripts for DSVA
 - `RSSDIVCS`: The dataset split and images of RSSDIVCS.
 - `flax-community`: The pretrained models：clip-rsicd and clip-rsicd-v2.



To perform evaluation, please download the model, then run ./script/XXXX.sh. 
For example, if you want to evaluate the model with the setting of GZSL and the split of 50/20, you should run ../script/12_gzsl_52_9.sh

## Results
Results from re-running models with this repo compared to reported numbers:
| **ZSL**   | **60/10** | **50/20** | **40/30** |
|---------------|------------|---------------|-----------|
| paper | 84.0 | 64.2 | 60.2 |
| repo | 86.6 | 63.8 | 60.4 |


| **GZSL**       | **U** | **S** | **H** |
|---------------|------------|---------------|-----------|
| 60/10 (paper) | 68.4 | 67.1 | 67.7 |
| 60/10 (repo) | 63.1 | 69.2 | 66.0 |
| 50/20 (paper) | 53.5 | 59.8 | 56.5 |
| 50/20 (repo) | 52.8 | 60.0 | 56.2 |
| 40/30 (paper) | 43.7 | 58.1 | 49.9 |
| 40/30 (repo) | 43.8 | 57.8 | 49.8 |

If you use any content of this repo for your work, please cite the following bib entry:

    @article{xu2023deep,
      title={Deep Semantic-Visual Alignment for zero-shot remote sensing image scene classification},
      author={Xu, Wenjia and Wang, Jiuniu and Wei, Zhiwei and Peng, Mugen and Wu, Yirong},
      journal={ISPRS Journal of Photogrammetry and Remote Sensing},
      volume={198},
      pages={140--152},
      year={2023},
      publisher={Elsevier}
    }

The code is under construction. If you have problems, feel free to reach me at xuwenjia@bupt.edu.cn

## Acknowledgment
This paper was supported in part by the National Key Research and Development Program of China under Grant 2021YFB2900200, National Natural Science Foundation of China under No. 61925101, and 61831002, 61921003, the Beijing Municipal Science and Technology, China Project NO. Z211100004421017


