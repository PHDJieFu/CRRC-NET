# Compact Representation and Reliable Classification Learning for Point-level Weakly-supervised Action Localization
### Official Pytorch Implementation of '[Compact Representation and Reliable Classification Learning for Point-level Weakly-supervised Action Localization](https://arxiv.org/abs/2108.05029)' (TIP 2022)

> **Compact Representation and Reliable Classification Learning for Point-level Weakly-supervised Action Localization**<br>
> Jie Fu, Junyu Gao and Changsheng Xu
>
> Paper: https://ieeexplore.ieee.org/abstract/document/9957128
>
> **Abstract:** *Point-level weakly-supervised temporal action localization (P-WSTAL) aims to localize temporal extents of action instances and identify the corresponding categories with only a single point label for each action instance for training. Due to the sparse frame-level annotations, most existing models are in the localization-by-classification pipeline. However, there exist two major issues in this pipeline: large intra-action variation due to task gap between classification and localization and noisy classification learning caused by unreliable pseudo training samples. In this paper, we propose a novel framework CRRC-Net, which introduces a co-supervised feature learning module and a probabilistic pseudo label mining module, to simultaneously address the above two issues. Specifically, the co-supervised feature learning module is applied to exploit the complementary information in different modalities for learning more compact feature representations. Furthermore, the probabilistic pseudo label mining module utilizes the feature distances from action prototypes to estimate the likelihood of pseudo samples and rectify their corresponding labels for more reliable classification learning. Comprehensive experiments are conducted on different benchmarks and the experimental results show that our method achieves favorable performance with the state-of-the-art.*


## Prerequisites
### Recommended Environment
* Python 3.6
* Pytorch 1.6
* Tensorflow 1.15 (for Tensorboard)
* CUDA 10.2

### Depencencies
You can set up the environments by using `$ pip3 install -r requirements.txt`.

### Data Preparation
1. Prepare [THUMOS'14](https://www.crcv.ucf.edu/THUMOS14/) dataset.
    - We excluded three test videos (270, 1292, 1496) as previous work did.

2. Extract features with two-stream I3D networks
    - We recommend extracting features using [this repo](https://github.com/piergiaj/pytorch-i3d).
    - For convenience, we provide the features we used. You can find them [here](https://drive.google.com/file/d/1NqaDRo782bGZKo662I0rI_cvpDT67VQU/view?usp=sharing).
    
3. Place the features inside the `dataset` folder.
    - Please ensure the data structure is as below.
   
~~~~
├── dataset
   └── THUMOS14
       ├── gt.json
       ├── split_train.txt
       ├── split_test.txt
       ├── fps_dict.json
       ├── point_gaussian
           └── point_labels.csv
       └── features
           ├── train
               ├── rgb
                   ├── video_validation_0000051.npy
                   ├── video_validation_0000052.npy
                   └── ...
               └── flow
                   ├── video_validation_0000051.npy
                   ├── video_validation_0000052.npy
                   └── ...
           └── test
               ├── rgb
                   ├── video_test_0000004.npy
                   ├── video_test_0000006.npy
                   └── ...
               └── flow
                   ├── video_test_0000004.npy
                   ├── video_test_0000006.npy
                   └── ...
~~~~

## Usage

### Running
You can easily train and evaluate the model by running the script below.

If you want to try other training options, please refer to `options.py`.

~~~~
$ bash run.sh
~~~~

### Evaulation

You can evaluate the model by running the command below.

~~~~
$ bash run_eval.sh
~~~~

## References
We note that this repo was built upon our previous models.
* Background Suppression Network for Weakly-supervised Temporal Action Localization (AAAI 2020) [[paper](https://arxiv.org/abs/1911.09963)] [[code](https://github.com/Pilhyeon/BaSNet-pytorch)]
* Weakly-supervised Temporal Action Localization by Uncertainty Modeling (AAAI 2021) [[paper](https://arxiv.org/abs/2006.07006)] [[code](https://github.com/Pilhyeon/WTAL-Uncertainty-Modeling)]
* Learning Action Completeness from Points for Weakly-supervised Temporal Action Localization [[paper](https://arxiv.org/abs/2108.05029)] [[code](https://github.com/Pilhyeon/Learning-Action-Completeness-from-Points)]

We referenced the repos below for the code.

* [STPN](https://github.com/bellos1203/STPN)
* [SF-Net](https://github.com/Flowerfan/SF-Net)
* [LAC](https://github.com/Pilhyeon/Learning-Action-Completeness-from-Points)
* [ActivityNet](https://github.com/activitynet/ActivityNet)

## Citation
If you find this code useful, please cite our paper.

~~~~
@article{fu2022compact,
  title={Compact Representation and Reliable Classification Learning for Point-Level Weakly-Supervised Action Localization},
  author={Fu, Jie and Gao, Junyu and Xu, Changsheng},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={7363--7377},
  year={2022},
  publisher={IEEE}
}
~~~~
