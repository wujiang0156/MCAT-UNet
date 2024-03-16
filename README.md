## Introduction

**MCAT-UNet** is an open-source semantic segmentation model based on mmsegmetation, 
which mainly focuses on developing advanced remote sensing image segmentation.

The proposed MCAT-UNet can extract local representations and capture long-range spatial dependencies to segment 
geographic objects more efficiently in complex scenarios with low computational complexity. In particular,
MCAT-UNet achieves more complete predictions for large-scale varied objects and small discrete multiscale objects, 
where the boundaries remain accurate and smooth. 
 
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


## Results and Models

### LoveDA/Potsdam/Vaihingen

| Method | Crop Size | Lr Schd | mIoU | #params(Mb) | FLOPs(Gbps) | config | log |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MACT-UNet | 512x512 | 100K | 75.44 | 23.2 | 18.5 | [config](configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py) | [github](https://github.com/wujiang0156/MCAT-UNet/releases/download/LOG/20231219_003217.log)
| MACT-UNet | 512x512 | 100K | 47.64 | xxx | xxx | [config](configs/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_small_patch4_window7_512x512.log.json)
| MACT-UNet | 512x512 | 100K | 48.13 | xxx | xxx | [config](configs/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_base_patch4_window7_512x512.log.json)


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
