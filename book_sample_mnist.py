import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
'''
《TensorFlow：实战Google深度学习框架（第二版）》书中第二个完整案例，位于第五章
本案例为完整的用来解决MNIST手写体数字的识别问题，单层
'''
# mnist数据集相关的常数

INPUT_NODE = 784   # 输入层节点数。对于mnist数据集，这个等于图片的像素
OUTPUT_NODE = 10   # 输出层的节点数。等于类别数目，0-9，共10个

# 配置神经网络的参数

LAYER1_NODE = 500  # 隐藏层节点数。这里使用单层隐藏层
BATCH_SIZE = 100   # 一个训练batch中的训练数据个数，数字越小时，训练过程月接近随机梯度下降；数字越大越接近梯度下降
LEARNING_RATE_BASE = 0.8    # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率

