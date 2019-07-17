# DPGANs
- [Single Image De-raining Using Detail Perceptual Generative Adversarial Networks]()
In this paper, Detail Perceptual Generative Adversarial Networks (DP-GANs) is proposed for single image de-raining. It consists of two convolutional neural networks (CNN): the rain streaks generative network G and the discriminative network D. To reduce the background interference, a rain streaks generative network is proposed, which not only focuses on the high frequency detail map of rainy image, but also directly reduces the mapping range from input to output. To further improve the perceptual quality of generated images, the perceptual loss is modified by extracting high-level features from discriminative network D, rather than pre-trained networks. Extensive
experiments on the synthetic and real-world datasets demonstrate that the proposed method achieves significant improvements over the recent state-of-the-art methods.
# Install
git clone https://github.com/CKZfd/DPGANs or Download it.

# Dataset
You can get the raw dataset from [here](https://pan.baidu.com/s/1Hvm9ctniC7PMQdKrI_lf3Q)
We use the dataset from [Removing rain from single images via a deep detail network](https://xueyangfu.github.io/projects/cvpr2017.html)
You need download it can copy it to the paired-dataset, you need change the path in the training_data.py and testing_data.py
run python training_data.py and python testing_data.py to generate our dataset and put the generated dataset in the facades

# Train
python main_DPGANs.py --dataroot ./facades/rainydataset/training --valDataroot ./facades/rainydataset/val

# Test



# Reference

# misc.

