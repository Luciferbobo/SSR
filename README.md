# High-Quality Real-Time Rendering Using Subpixel Sampling Reconstruction
[Project Page](https://thatbobo.com/SSR.github.io/) | [Paper](https://arxiv.org/abs/2301.01036) 

[_AAAI 2024_] High-Quality Real-Time Rendering Using Subpixel Sampling Reconstruction




<div align=center>

<img src="https://github.com/Luciferbobo/SSR/blob/main/images/tf.png" width="1000"> 

</div>

## Abstract

Generating high-quality, realistic rendering images for real-time applications generally requires tracing a few samples-per-pixel (spp) and using deep learning-based approaches to denoise the resulting low-spp images. Existing denoising methods have yet to achieve real-time performance at high resolutions due to the physically-based sampling and network inference time costs. In this paper, we propose a novel Monte Carlo sampling strategy to accelerate the sampling process and a corresponding denoiser, subpixel sampling reconstruction (SSR), to obtain high-quality images. Extensive experiments demonstrate that our method significantly outperforms previous approaches in denoising quality and reduces overall time costs, enabling real-time rendering capabilities at 2K resolution.

## Data Preparation

Please download [SSR dataset](https://pan.baidu.com/s/1rwoE82xNisf--xBD5mwjUg?pwd=ssr8) and organize the data as follows:

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
