## MCAT-UNet

**MCAT-UNet** is an open-source semantic segmentation model based on mmsegmetation, 
which mainly focuses on developing advanced remote sensing image segmentation.
Article download link https://ieeexplore.ieee.org/abstract/document/10521698

The proposed MCAT-UNet can extract local representations and capture long-range spatial dependencies to segment 
geographic objects more efficiently in complex scenarios with low computational complexity. In particular,
MCAT-UNet achieves more complete predictions for large-scale varied objects and small discrete multiscale objects, 
where the boundaries remain accurate and smooth. 
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
