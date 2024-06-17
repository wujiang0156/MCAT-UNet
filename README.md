## Introduction

Remote sensing change detection (RSCD), which aims to identify differences between bitemporal images, has made great progress through the application of deep learning methods. Convolutional neural networks (CNNs) and transformers are widely used in remote sensing image change detection and have achieved promising results. However, current models predominantly focus on visual representation learning while neglecting the potential of multimodal learning methods. Consequently, this leads to issues such as inaccurate identification of nonsemantic changes, incomplete boundary extraction due to the degradation of local feature details, and the loss of small targets. Recently, a novel method for efficiently learning from natural language supervision, known as contrastive language image pretraining (CLIP), has been proposed. This innovative paradigm demonstrates impressive performance on downstream tasks. In this work, we propose a new change detection framework, CLIPFormer, which leverages pretraining knowledge from CLIP and the Swin transformer. Specifically, we reconstruct the original CLIP architecture using the Swin transformer to extract bitemporal features and introduce a novel difference feature enhancement module to capture detailed semantic change features. Additionally, in the decoding stage, we propose a vision language guided transformer (VLGT) multimodal decoder by integrating cross-attention with the cross-shaped window transformer (CSWT) module to enhance the image semantic representation. Extensive experiments demonstrate that our model outperforms other state-of-the-art (SOTA) methods, with IoU values of 85.60%, 76.81%, 89.55%, 94.51%, and 71.77% on the LEVIR-CD, LEVIR-CD+, WHUCD, CDD, and SYSU-CD datasets, respectively. The source code for this work is available at https://github.com/wujiang0156/CLIPFormer
 
## Install
- First, you need to download mmsegmentation and install it on your server.
- Second, place backbone.py and csheadunet.py in the corresponding directory of mmsegmentation.
- Third, train according to the training strategy of mmsegmentation and the training parameters in our paper.

## Pretrained Weights of Backbones

[pretrain](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_s_20230227-f33ccdf2.pth)

## Data Preprocessing

Download the datasets from the official website and split them yourself.

**Potsdam and Vaihingen**
[Potsdam and Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)

**LoveDA**
[LoveDA](https://github.com/Junjue-Wang/LoveDA)

## Training

You can refer to **mmsegmentation document** (https://mmsegmentation.readthedocs.io/en/latest/index.html).


## Results and Logs for MACT-UNet

### LoveDA/Potsdam/Vaihingen

| Dataset | Crop Size | Lr Schd | mIoU | #params(Mb) | FLOPs(Gbps) | config | log |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Potsdam | 512x512 | 100K | 75.44 | 23.2 | 18.5 | [config](https://github.com/wujiang0156/MCAT-UNet/blob/main/config/segnext_csheadunet-s_1xb4-adamw-100k_potsdam-512x512.py) | [github](https://github.com/wujiang0156/MCAT-UNet/releases/download/LOG/20231219_003217.log)
| Vaihingen | 512x512 | 100K | 74.52 |  23.2 | 18.5 | [config](https://github.com/wujiang0156/MCAT-UNet/blob/main/config/segnext_csheadunet-s_1xb4-adamw-100k_vaihingen-512x512.py) | [github](https://github.com/wujiang0156/MCAT-UNet/releases/download/LOG/20240106_074357.log)
| LoveDa | 512x512 | 100K | 53.58 |  23.2 | 18.5 | [config](https://github.com/wujiang0156/MCAT-UNet/blob/main/config/segnext_csheadunet-s_1xb4-adamw-100k_loveda-512x512.py) | [github](https://github.com/wujiang0156/MCAT-UNet/releases/download/LOG/20231226_030251.log)


## Inference on High-resolution remote sensing image

<div>
<img src="fig 6.jpg" width="100%"/>
<img src="fig 7.jpg" width="100%"/>
<img src="fig 8.jpg" width="100%"/>
<img src="fig 9.jpg" width="100%"/>
</div>

## Acknowledgement

 Many thanks the following projects's contributions to **MACT-UNet**.
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
