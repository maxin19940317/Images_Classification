from __future__ import print_function
import cv2 as cv
import os
import keras
from keras.preprocessing.image import img_to_array
import numpy as np

import tensorflow as tf

config1 = tf.ConfigProto()
config1.gpu_options.allow_growth = True
tf.Session(config=config1)


def get_class_list():
    with open('class_idx.txt', 'r') as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]  # rstrip 默认删除字符串后的空格
    return lines


def load_model(model_path):
    model = keras.applications.ResNet50(include_top=True,
                                        weights=None,
                                        input_tensor=None,
                                        input_shape=(224, 224, 3),
                                        pooling='max',
                                        classes=14)
    model.load_weights(model_path)
    return model


def predict(model, img_test):
    img = cv.resize(img_test, (224, 224), interpolation=cv.INTER_LINEAR)

    img = np.array([img_to_array(img)], dtype='float') / 255.0  # 归一化 加快网络收敛性

    label_list = get_class_list()
    pred = model.predict(img)
    index = np.argmax(pred)
    label = label_list[int(index)]
    confidence = np.max(pred)

    return label, confidence


if __name__ == '__main__':
    model = load_model(model_path='./models/weights.14-0.06.h5')

    test_dir = '/media/cyg/DATA1/DataSet/classifier/dataset/test/'
    sub_dir_list = os.listdir(test_dir)

    for sub_dir in sub_dir_list:
        print(sub_dir)
        img_test_list = os.listdir(os.path.join(test_dir, sub_dir))
        error_num = 0
        for image_name in img_test_list:
            image_path = os.path.join(test_dir, sub_dir, image_name)
            img_test = cv.imread(image_path)
            label, confidence = predict(model=model, img_test=img_test)
            if label != sub_dir:
                error_num = error_num + 1
        print('label: {} acc: {}'.format(sub_dir, 1 - error_num / len(img_test_list)))
