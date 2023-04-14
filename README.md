# QuantArt
Official PyTorch implementation of the paper:

[**QuantArt: Quantizing Image Style Transfer Towards High Visual Fidelity**](https://arxiv.org/abs/2212.10431)  
[Siyu Huang<sup>*</sup>](https://siyuhuang.github.io), [Jie An<sup>*</sup>](https://www.cs.rochester.edu/u/jan6/), [Donglai Wei](https://donglaiw.github.io/), [Jiebo Luo](https://www.cs.rochester.edu/u/jluo/), [Hanspeter Pfister](https://vcg.seas.harvard.edu/people/hanspeter-pfister)  
CVPR 2023

We devise a new style transfer framework called QuantArt for high visual-fidelity stylization. The core idea is pushing the latent representation of the generated artwork toward the centroids of the real artwork distribution with vector quantization. QuantArt achieves decent performance for various image style transfer tasks.

<p align='center'>
 <img alt='Thumbnail' src='imgs/thumb.png'>
</p>

## Dependencies
* python=3.8.5
* pytorch=1.7.0
* pytorch-lightning=1.0.8
* cuda=10.2
 
We recommend to use `conda` to create a new environment with all dependencies installed.
```
conda env create -f environment.yaml
conda activate quantart
```

## Datasets and Pre-trained Models
**Stage-1:** The datasets and models for codebook pretraining in this repository are as follows:

| Dataset | Pre-trained Model |
| ---- | ---- |
| [MS_COCO](https://cocodataset.org/#download) | [vqgan_imagenet_f16_1024.ckpt](https://drive.google.com/file/d/1lcrBplMVQTO6-ppxSWUyD_2coUiUpwoS/view?usp=share_link) |
| [WikiArt](https://www.kaggle.com/competitions/painter-by-numbers/data) | [vqgan_wikiart_f16_1024.ckpt](https://drive.google.com/file/d/1xIYbaXLEdroYeftzM_1r5q2P9ANhQQpv/view?usp=share_link) |
| [LandscapesHQ](https://github.com/universome/alis) | [vqgan_landscape_f16_1024.ckpt](https://drive.google.com/file/d/13VjJonTCJWz2QEIGX_KeO1RB3t15qTBE/view?usp=share_link) |
| [FFHQ](https://github.com/NVlabs/ffhq-dataset) | [vqgan_faceshq_f16_1024.ckpt](https://drive.google.com/file/d/1_6ZW8iVhFentkG_HTn5pMj4JRAiPUHwY/view?usp=share_link) |
| [Metfaces](https://github.com/NVlabs/metfaces-dataset) | [vqgan_metfaces_f16_1024.ckpt](https://drive.google.com/file/d/1omGG6TmSVsksk39pGPkyLVTa-K4FcxQd/view?usp=share_link) |

**Stage-2:** The datasets and models for style transfer experiments in this repository are as follows:

| Task | Pre-trained Model | Content | Style |
| ---- | ---- | ---- | ---- |
| photo->artwork| [coco2art](https://drive.google.com/drive/folders/13-z3eowPsPjKTIULP5sBrr_jgJn3w7ZH?usp=share_link) | [MS_COCO](https://cocodataset.org/#download) | [WikiArt](https://www.kaggle.com/competitions/painter-by-numbers/data) | 
| landscape->artwork | [landscape2art](https://drive.google.com/drive/folders/1zuz9CmgpB7JsEx-Y5H0K0u3D95C6g4MU?usp=share_link)| [LandscapesHQ](https://github.com/universome/alis) |[WikiArt](https://www.kaggle.com/competitions/painter-by-numbers/data) |
| landscape->artwork (non-VQ) | [landscape2art_continuous](https://drive.google.com/drive/folders/1s-N62W8l_1iOvydsWvmJTWxNwWWJKTum?usp=share_link)| [LandscapesHQ](https://github.com/universome/alis) | [WikiArt](https://www.kaggle.com/competitions/painter-by-numbers/data) | [WikiArt](https://www.kaggle.com/competitions/painter-by-numbers/data) |
| face->artwork | [face2art](https://drive.google.com/drive/folders/1wKWmmtLChtXTWFaaun097H7lYJ6-IWTe?usp=share_link) | [FFHQ](https://github.com/NVlabs/ffhq-dataset) | [Metfaces](https://github.com/NVlabs/metfaces-dataset) |
| artwork->artwork | [art2art](https://drive.google.com/drive/folders/1J48c21bN5f9anGBUSALcEMO0Bj8Emd0s?usp=share_link) | [WikiArt](https://www.kaggle.com/competitions/painter-by-numbers/data) | [WikiArt](https://www.kaggle.com/competitions/painter-by-numbers/data) |
| photo->photo | [coco2coco](https://drive.google.com/drive/folders/1xc5P1woZJSoemcVvjnZ4Gj5Jyam7RsQZ?usp=share_link) | [MS_COCO](https://cocodataset.org/#download) | [MS_COCO](https://cocodataset.org/#download) |
| landscape->landscape | [landscape2landscape](https://drive.google.com/drive/folders/1bmL25tOwuXt63wXwpNwlSW775sjrPhxL?usp=share_link) | [LandscapesHQ](https://github.com/universome/alis) | [LandscapesHQ](https://github.com/universome/alis) |

## Quick Start
1. To test the landscape-to-artwork style transfer performance, please download the [LandscapesHQ](https://disk.yandex.ru/d/Sz1gPiMoUregEQ) and [WikiArt](https://www.kaggle.com/competitions/painter-by-numbers/data) datasets and put them under `./datasets/`. 

2. Download the pre-trained [landscape2art](https://drive.google.com/drive/folders/1zuz9CmgpB7JsEx-Y5H0K0u3D95C6g4MU?usp=share_link) model and put it under `./logs/`.  The folder structure should be:

```
├── configs
├── datasets
│   ├── lhq_1024_jpg
│   │   ├── lhq_1024_jpg
│   │   │   ├── 0000000.jpg
│   │   │   ├── 0000001.jpg
│   │   │   ├── 0000002.jpg
│   │   │   ├── ...
│   ├── painter-by-numbers
│   │   ├── train
│   │   │   ├── 100001.jpg
│   │   │   ├── 100002.jpg
│   │   │   ├── 100003.jpg
│   │   │   ├── ...
│   │   ├── test
│   │   │   ├── 0.jpg
│   │   │   ├── 100000.jpg
│   │   │   ├── 100004.jpg
│   │   │   ├── ...
├── logs
│   ├── landscape2art
│   │   ├── checkpoints
│   │   ├── configs
├── taming
├── environment.yaml
├── main.py
├── train.sh
└── test.sh
```

3. Run the following command:
```
bash test.sh
```
The landscape-to-artwork style transfer results will be saved in `./logs/`.

## Training
Run `bash train.sh` or the following command to train a photo-to-artwork model
```
python -u main.py --base configs/landscape2art.yaml -t True --gpus 0,
```

* `--base`: path for the config file.
* `-t`: is training.


## Citation
```
@inproceedings{huang2023quantart,
    title={QuantArt: Quantizing Image Style Transfer Towards High Visual Fidelity},
    author={Siyu Huang and Jie An and Donglai Wei and Jiebo Luo and Hanspeter Pfister},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    month={June},
    year={2023}
}
```

## Acknowledgement
This repository is heavily built upon [VQGAN](https://github.com/CompVis/taming-transformers).

## Contact
If you have any questions, please do not hesitate to contact <huangsiyutc@gmail.com>.
