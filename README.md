## Introduction

**MCAT-UNet** is an open-source semantic segmentation model based on mmsegmetation, 
which mainly focuses on developing advanced remote sensing image segmentation.

The proposed MCAT-UNet can extract local representations and capture long-range spatial dependencies 
to segment geographic objects more efficiently in complex scenarios with low computational complexity.
 
## Install
- First, you need to download mmsegmentation and install it on your server.
- Second, place backbone.py and csheadunet.py in the corresponding directory of mmsegmentation.
- Third, train according to the training strategy of mmsegmentation and the training parameters in our paper.

## Pretrained Weights of Backbones

[Google Drive](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_s_20230227-f33ccdf2.pth)

## Data Preprocessing

Download the datasets from the official website and split them yourself.

**Potsdam and Vaihingen**
[Potsdam and Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)

**LoveDA**
[LoveDA](https://github.com/Junjue-Wang/LoveDA)

## Training

You can refer to **mmsegmentation document** (https://mmsegmentation.readthedocs.io/en/latest/index.html).

## Inference on huge remote sensing image

<div>
<img src="fig 6.jpg" width="100%"/>
<img src="fig 7.jpg" width="100%"/>
<img src="fig 8.jpg" width="100%"/>
<img src="fig 9.jpg" width="100%"/>
</div>

## Acknowledgement

 Many thanks the following projects's contributions to **MACT-UNet**.
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
