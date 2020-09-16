"""
预测脚本
"""

import os
import tqdm
import pandas as pd
import cv2 as cv
import numpy as np
import keras
from keras.preprocessing.image import img_to_array
import tensorflow as tf

config1 = tf.ConfigProto()
config1.gpu_options.allow_growth = True
tf.Session(config=config1)


def load_model(model_path):
    """
    加载模型
    :param model_path: 模型所在路径
    :return: 返回带有权重的模型
    """
    model = keras.applications.ResNet50(include_top=True,
                                        weights=None,
                                        input_tensor=None,
                                        input_shape=(224, 224, 3),
                                        pooling='max',
                                        classes=14)
    model.load_weights(model_path)
    return model


def predict(model, img_test):
    """
    预测单张图像分类
    :param model: 加载权重的分类模型
    :param img_test: 输入图像(BGR)
    :return: 返回预测分类的索引和置信度
    """
    img_test = cv.resize(img_test, (224, 224), interpolation=cv.INTER_LINEAR)
    img_test = np.array([img_to_array(img_test)], dtype='float') / 255.0

    y_pried = model.predict(img_test)

    index = np.argmax(y_pried)
    confidence = np.max(y_pried)

    return index, confidence


if __name__ == '__main__':
    model_path = './models/models.12-0.01-0.995.h5'
    if not os.path.isfile(model_path):
        raise FileExistsError('{} is not exist!'.format(model_path))
    model = load_model(model_path=model_path)

    test_data = pd.read_csv('data/test.csv')
    label_index_list = test_data['label_index']
    error_num = 0
    for index, image_path in enumerate(tqdm.tqdm(test_data['image_path'])):
        img = cv.imread(image_path)
        label_index, confidence = predict(model, img)
        if label_index != label_index_list[index]:
            error_num = error_num + 1
            print(image_path, label_index_list[index], label_index, confidence)
    print('acc: {}'.format(1 - error_num / len(test_data['image_path'])))
