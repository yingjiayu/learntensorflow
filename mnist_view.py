from tensorflow.examples.tutorials.mnist import input_data
import math
import matplotlib.pyplot as plt
import numpy as np
mnist = input_data.read_data_sets('/tmp/data',one_hot=True)
#画单张mnist数据集的数据
def drawdigit(position,image,title):
    plt.subplot(*position)
    plt.imshow(image,cmap='gray_r')
    plt.axis('off')
    plt.title(title)

#取一个batch的数据，然后在一张画布上画batch_size个子图
def batchDraw(batch_size):
    images,labels = mnist.train.next_batch(batch_size)
    row_num = math.ceil(batch_size ** 0.5)
    column_num = row_num
    plt.figure(figsize=(row_num,column_num))
    for i in range(row_num):
        for j in range(column_num):
            index = i * column_num + j
            if index < batch_size:
                position = (row_num,column_num,index+1)
                image = images[index].reshape(-1,28)
                title = 'actual:%d'%(np.argmax(labels[index]))
                drawdigit(position,image,title)


if __name__ == '__main__':
    batchDraw(324)
    plt.show()