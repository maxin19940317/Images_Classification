"""
ResNet训练脚本
"""

import os
import yaml
import pandas as pd
import tqdm

import cv2 as cv
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

config1 = tf.ConfigProto()
config1.gpu_options.allow_growth = True
tf.Session(config=config1)


class Model(object):
    """
    构建Model类
    """
    def __init__(self, config):
        self.train_images_dir = config['TRAIN_DATA_DIR']
        self.checkpoints = config['MODEL_DIR']
        self.im_size = config['IMAGE_SIZE']
        self.epochs = config['EPOCHS']
        self.batch_size = config['BATCH_SIZE']
        self.classNumber = config['CLASS_NUM']
        self.lr = config['LEARNING_RATE']
        self.data_augmentation = config['IS_AUGMENTATION']
        self.model = self.build_model()

    def build_model(self):
        model = keras.applications.ResNet50(include_top=True,
                                            weights=None,
                                            input_tensor=None,
                                            input_shape=(self.im_size, self.im_size, 3),
                                            pooling='max',
                                            classes=self.classNumber)
        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(lr=self.lr),
                      metrics=["accuracy"])
        return model

    def load_data(self):
        """
        功能：加载训练数据数据
        :return: 训练集和测试集按9:1划分
        """
        train_data = pd.read_csv('data/train.csv')
        images_data = []

        for image_path in tqdm.tqdm(train_data['image_path']):
            img = cv.imread(image_path)
            img = cv.resize(src=img, dsize=(self.im_size, self.im_size), interpolation=cv.INTER_LINEAR)
            images_data.append(img)
        images_data = np.array(images_data, dtype='float32') / 255.0
        label_index_list = train_data['label_index']

        labels = to_categorical(np.array(label_index_list), num_classes=self.classNumber)
        x_train, x_test, y_train, y_test = train_test_split(images_data, labels, test_size=0.1, stratify=labels)  # 划分数据集为训练集和测试集 比例为9：1
        return x_train, x_test, y_train, y_test

    def train(self):
        os.makedirs(config['MODEL_DIR'], exist_ok=True)

        lr_reduce = keras.callbacks.ReduceLROnPlateau(monitor=config['MONITOR'],
                                                      factor=0.5,  # 学习率下降系数
                                                      patience=config['LR_REDUCE_PATIENCE'],
                                                      verbose=1,
                                                      mode='auto',
                                                      cooldown=0)
        early_stop = keras.callbacks.EarlyStopping(monitor=config['MONITOR'],  # 当监测值，该回调函数将中止训练
                                                   min_delta=0,
                                                   patience=config['EARLY_STOP_PATIENCE'],  # 提前终止训练的步长
                                                   verbose=1,
                                                   mode='auto')
        checkpoint = keras.callbacks.ModelCheckpoint(filepath=os.path.join(config['MODEL_DIR'], 'models.{epoch:02d}-{val_loss:.5f}-{val_acc:.5f}.h5'),
                                                     monitor=config['MONITOR'],
                                                     verbose=1,
                                                     save_best_only=True,
                                                     save_weights_only=True,
                                                     mode='auto',
                                                     period=1)

        x_train, x_test, y_train, y_test = self.load_data()
        if self.data_augmentation:
            data_aug = ImageDataGenerator(rotation_range=5,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          zoom_range=0.3,
                                          horizontal_flip=False)
            data_aug.fit(x_train)
            self.model.fit_generator(generator=data_aug.flow(x_train, y_train, batch_size=self.batch_size),
                                     steps_per_epoch=x_train.shape[0] // self.batch_size,
                                     validation_data=(x_test, y_test),
                                     shuffle=True,
                                     epochs=self.epochs, verbose=1, max_queue_size=1000,
                                     callbacks=[early_stop, checkpoint, lr_reduce])
        else:
            self.model.fit(x=x_train, y=y_train,
                           batch_size=self.batch_size,
                           validation_data=(x_test, y_test),
                           epochs=self.epochs,
                           callbacks=[early_stop, checkpoint, lr_reduce],
                           shuffle=True,
                           verbose=1)


if __name__ == '__main__':
    config_path = './configs/config.yaml'
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = Model(config)
    model.train()  # 开始训练模型
