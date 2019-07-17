# DPGANs
- [Single Image De-raining Using Detail Perceptual Generative Adversarial Networks]()
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

