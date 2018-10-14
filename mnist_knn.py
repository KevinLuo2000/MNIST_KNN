import numpy as np
import time
TRAINING_LABELS = 'train-labels-idx1-ubyte'
'''
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
0004     32 bit integer  60000            number of items 
0008     unsigned byte   ??               label 
0009     unsigned byte   ??               label 
........ 
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.
'''
TRAINING_IMAGES = 'train-images-idx3-ubyte'
'''
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number 
0004     32 bit integer  60000            number of images 
0008     32 bit integer  28               number of rows 
0012     32 bit integer  28               number of columns 
0016     unsigned byte   ??               pixel 
0017     unsigned byte   ??               pixel 
........ 
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
'''
TEST_IMAGES = 't10k-images-idx3-ubyte'
'''
TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number 
0004     32 bit integer  10000            number of images 
0008     32 bit integer  28               number of rows 
0012     32 bit integer  28               number of columns 
0016     unsigned byte   ??               pixel 
0017     unsigned byte   ??               pixel 
........ 
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black). 
'''
TEST_LABELS = 't10k-labels-idx1-ubyte'
'''
TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
0004     32 bit integer  10000            number of items 
0008     unsigned byte   ??               label 
0009     unsigned byte   ??               label 
........ 
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.
'''


# 读取32位数字 8位（bit） = 1字节（byte），32位即是4字节
def read32bit(file):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(file.read(4), dtype=dt)[0]


def extract_images(images):
    # rb读二进制图片文件，函数执行一次该文件内所有图片都读完
    with open(images, 'rb') as images:
        # 读image集前四个元素
        magic_number = read32bit(images)
        number_of_images = read32bit(images)
        number_of_rows = read32bit(images)
        number_of_columns = read32bit(images)
        # 英文明确一下，row是行，横着的，column是列，竖着的
        print("魔法值："+str(magic_number), "图片数"+str(number_of_images), "行数"+str(number_of_rows),
              "列数"+str(number_of_columns))
        # 接下来呢，8位8位地读，灰度从十进制的0到255，而256=2^8,8位二进制数正好能表示。把数据读个遍
        data = np.frombuffer(images.read(number_of_columns*number_of_rows*number_of_images), dtype=np.uint8)
        # 读完以后不能就这么放着呀，把它转换成矩阵吧，看着方便点，reshape一下
        # 每个图片占一行，共60000个文件也就是60000行，或者理解为60000个一维数组，输出第n个图片：print(data[n-1])
        data = data.reshape(number_of_images, number_of_rows * number_of_columns)
        # 记得用.shape看行、列数
        # reshape成矩阵这样写,相当于60000个二维数组，每个二维数组28*28,输出第n个图片：print(data[n-1]),输出第n个图片a行b列
        # data = data.reshape(number_of_images, number_of_rows, number_of_columns)
        # 灰度值二值化，不是0的都是1
        # 超级简洁啊，不用for循环表示data下标，直接整体里每一项和1比大小然后return
        # 还有一点，一般思维是if not data[][][] == 0, data[][][] = 1，这里直接取minimum，很有意思
        return np.minimum(data, 1)


def extract_labels(labels):
    with open(labels, 'rb') as labels:
        magic_number = read32bit(labels)
        number_of_items = read32bit(labels)
        print("魔法值："+str(magic_number), "标签个数"+str(number_of_items))
        data = np.frombuffer(labels.read(number_of_items), dtype=np.uint8)
        return data


def knn_algorithm(test, training, training_label, parameter):
    training_number = training.shape[0]
    # 把test沿x轴方向复制training_number遍(覆盖，故复制一遍相当于原来模样没变)，y轴方向复制一遍
    test_copy = np.tile(test, (training_number, 1))
    # 类似标准差，只不过没有了/n的操作
    comparison = (test_copy - training)**2
    # 把一行的加起来
    distance2 = np.sum(comparison, axis=1)
    # 开根号
    distance = distance2 ** 0.5
    # distance从小到大排，存它们的索引
    sorted_distance_indices = np.argsort(distance)
    # return print(distance[:100])
    count = {}
    for i in range(parameter):
        # 参数的选择异常重要，选大了跑的时间长而成效甚微，选小了频率与概率值偏差大，容易出问题->我这里比你黑一点，那里比你白一点
        # 最后我俩distance差不多
        label = training_label[sorted_distance_indices[i]]
        # 又避免了一个if else
        count[label] = count.get(label, 0)+1
    max_value = 0
    max_key = 0
    for key, value in count.items():
        if value > max_value:
            max_value = value
            max_key = key
    return max_key


def test_session(test_amount):
    start = time.clock()
    training_labels = extract_labels(TRAINING_LABELS)
    training_images = extract_images(TRAINING_IMAGES)
    test_labels = extract_labels(TEST_LABELS)
    test_images = extract_images(TEST_IMAGES)
    counter = 0
    # TEST_IMAGES.shape[0]
    for i in range(test_amount):
        key = knn_algorithm(test_images[i], training_images, training_labels, 10)
        if key == test_labels[i]:
            counter = counter+1
    rate = counter / test_amount * 100
    end = time.clock()
    return print("正确率："+str(rate)+"%，用时"+str(end-start)+"秒")


if __name__ == '__main__':
    test_session(10000)
