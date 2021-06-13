# LARGE: Latent-Based Regression through GAN Semantics

<p align="center">
<img src="docs/imgs/pipeline_fig.png" width="800px"/>
</p>

## [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/YotamNitzan/LARGE/blob/main/latent_regression.ipynb)

<!-- ### [[Paper](https://arxiv.org/abs/2004.15021)]-->
### [[Project Website](https://yotamnitzan.github.io/LARGE/)] [[Google Colab](http://colab.research.google.com/github/YotamNitzan/LARGE/blob/main/latent_regression.ipynb)]

LARGE: Latent-Based Regression through GAN Semantics

[Yotam Nitzan<sup>*</sup>](https://yotamnitzan.github.io), [Rinon Gal<sup>*</sup>](https://rinongal.github.io/), [Ofir Brenner](https://scholar.google.com/citations?user=iLLlWr8AAAAJ), and [Daniel Cohen-Or](https://danielcohenor.com/)

Abstract: We propose a novel method for solving regression tasks using few-shot or weaksupervision. At the core of our method is the fundamental observation that GANsare incredibly successful at encoding semantic information within their latent space,even in a completely unsupervised setting. For modern generative frameworks,this semantic encoding manifests as smooth, linear directions which affect imageattributes in a disentangled manner. These directions have been widely used inGAN-based image editing. We show that such directions are not only linear, but thatthe magnitude of change induced on the respective attribute is approximately linearwith respect to the distance traveled along them. By leveraging this observation,our method turns a pre-trained GAN into a regression model, using as few as twolabeled samples. This enables solving regression tasks on datasets and attributeswhich are difficult to produce quality supervision for. Additionally, we showthat the same latent-distances can be used to sort collections of images by thestrength of given attributes, even in the absence of explicit supervision. Extensiveexperimental evaluations demonstrate that our method can be applied across a widerange of domains, leverage multiple latent direction discovery frameworks, andachieve state-of-the-art results in few-shot and low-supervision settings, even whencompared to methods designed to tackle a single task

<!-- 
## Citation

If you find our code useful, please consider citing our paper:

```
@article{Nitzan2020FaceID,
  title={Face identity disentanglement via latent space mapping},
  author={Yotam Nitzan and A. Bermano and Yangyan Li and D. Cohen-Or},
  journal={ACM Transactions on Graphics (TOG)},
  year={2020},
  volume={39},
  pages={1 - 14}
}
```
-->

## Sorting Examples

<p align="center">
Black to Blond hair <br> <img src="docs/imgs/sort_black_blonde_hair.png" width="800px"/>
</p>
<p align="center">
Age <br> <img src="docs/imgs/sort_cat_age.jpg" width="800px"/> 
</p>
<p align="center">
Fur Fluffiness <br> <img src="docs/imgs/sort_dog_fluffiness.jpg" width="800px"/>
</p>
<p align="center">
Sickness <br> <img src="docs/imgs/rot.png" width="800px"/>
</p>

## Credits

**StyleGAN2 implementation:**  
https://github.com/rosinality/stylegan2-pytorch  
Copyright (c) 2019 Kim Seonghyeon  
License (MIT) https://github.com/rosinality/stylegan2-pytorch/blob/master/LICENSE 

**StyleGAN2 Models:**
https://github.com/NVlabs/stylegan2-ada/
https://github.com/NVlabs/stylegan2
Copyright (c) 2021, NVIDIA Corporation
[Nvidia Source Code License-NC](https://nvlabs.github.io/stylegan2/license.html)

**pSp model and implementation:**   
https://github.com/eladrich/pixel2style2pixel  
Copyright (c) 2020 Elad Richardson, Yuval Alaluf  
License (MIT) https://github.com/eladrich/pixel2style2pixel/blob/master/LICENSE

**e4e model and implementation:**   
https://github.com/omertov/encoder4editing
Copyright (c) 2021 omertov  
License (MIT) https://github.com/omertov/encoder4editing/blob/main/LICENSE

**ReStyle model and implementation:**   
https://github.com/yuval-alaluf/restyle-encoder/
Copyright (c) 2021 Yuval Alaluf  
License (MIT) https://github.com/yuval-alaluf/restyle-encoder/blob/main/LICENSE

## Acknowledgement

We would like to thank Raja Gyres, Yangyan Li, Or Patashnik, Yuval Alaluf, Amit Attia, Noga Bar and Zonzge Wu for helpful comments. We additionaly thank Zonzge Wu for the trained e4e models for AFHQ cats and dogs.
