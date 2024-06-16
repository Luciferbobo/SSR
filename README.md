# High-Quality Real-Time Rendering Using Subpixel Sampling Reconstruction
[Project Page](https://thatbobo.com/SSR.github.io/) | [Paper](https://arxiv.org/abs/2301.01036) 

[_AAAI 2024_] High-Quality Real-Time Rendering Using Subpixel Sampling Reconstruction




<div align=center>

<img src="https://github.com/Luciferbobo/SSR/blob/main/images/tf.png" width="1000"> 

</div>

## Abstract

Generating high-quality, realistic rendering images for real-time applications generally requires tracing a few samples-per-pixel (spp) and using deep learning-based approaches to denoise the resulting low-spp images. Existing denoising methods have yet to achieve real-time performance at high resolutions due to the physically-based sampling and network inference time costs. In this paper, we propose a novel Monte Carlo sampling strategy to accelerate the sampling process and a corresponding denoiser, subpixel sampling reconstruction (SSR), to obtain high-quality images. Extensive experiments demonstrate that our method significantly outperforms previous approaches in denoising quality and reduces overall time costs, enabling real-time rendering capabilities at 2K resolution.

## Preparation

This repo is tested with Ubuntu 20.04, python==3.7/3.8, pytorch==1.4.0 and cuda==10.1.

Please download [SSR dataset](https://pan.baidu.com/s/1rwoE82xNisf--xBD5mwjUg?pwd=ssr8) and organize the data as follows, then set path in the settings.py with the corresponding data location.

```
Subpixel dataset
├── spp32768_train
|  └── [scene name]
|  └── ...
├── spp32768_test
|  └── [scene name]
|  └── ...
├── spp32768_val
|  └── [scene name]
|  └── ...
...
```

Here we provide a detailed introduction to the G-buffer features.

<div align="center">

|          | R             | G         | B         | A               |
|----------|---------------|-----------|-----------|-----------------|
| Color    | albedo        |           |           |                 |
| Normal   | normal        |           |           | AlphaMode       |
| Position | position      |           |           | HitModelFlag    |
| Emissive | emissive      |           |           | AO              |
| PBR      | bDoubleSided  | roughness | metallic  | AlphaCutoff     |
| FWidth   | N Width       | depth     | position  | PrimitiveID     |
|          | R16           | G16       |           |                 |
| Velocity | x             | y         | ViewDist  | Mesh ID         |
| NDC      | x             | y         | z         | w               |

</div>



## Training & Evaluation

All training and hyperparameter settings are in setting.py.

Train SSR 
```
python3 train.py
```

Test with different best checkpints
```
python3 test.py --checkpoint psnr
python3 test.py --checkpoint ssim
python3 test.py --checkpoint rmse
```

## Baselines

We additionally provide baselines reproduction code:

[Monte Carlo Denoising via Auxiliary Feature Guided Self-Attention (TOG 2021)](https://aatr0x13.github.io/AFGSA.github.io/afgsa.html)

[Interactive Monte Carlo Denoising using Affinity of Neural Features (SIGGRAPH 2021)](https://www.mustafaisik.net/anf/)

[Neural Supersampling for Real-time Rendering (SIGGRAPH 2020)](https://research.facebook.com/publications/neural-supersampling-for-real-time-rendering/)

[Interactive Reconstruction of Monte Carlo Image Sequences using a Recurrent Denoising Autoencoder (SIGGRAPH 2017)](https://research.nvidia.com/publication/2017-07_interactive-reconstruction-monte-carlo-image-sequences-using-recurrent)


## Citing
```
@article{zhang2023high,
  title={High-Quality Real-Time Rendering Using Subpixel Sampling Reconstruction},
  author={Zhang, Boyu and Yuan, Hongliang and Zhu, Mingyan and Liu, Ligang and Wang, Jue},
  journal={arXiv preprint arXiv:2301.01036v2},
  year={2023}
}

or

@article{zhang2023high,
  title={High-Quality Supersampling via Mask-reinforced Deep Learning for Real-time Rendering},
  author={Zhang, Boyu and Yuan, Hongliang and Zhu, Mingyan and Liu, Ligang and Wang, Jue},
  journal={arXiv preprint arXiv:2301.01036v1},
  year={2023}
}
```
