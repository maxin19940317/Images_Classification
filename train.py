import cv2
import glob
import itertools
import os
import sys
import tqdm
from random import shuffle

import keras
import numpy as np
import tensorflow as tf
import yaml
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

config1 = tf.ConfigProto()
config1.gpu_options.allow_growth = True
tf.Session(config=config1)

# sys.setrecursionlimit(10000)


class Build_Model(object):
    def __init__(self, config):
        self.train_images_dir = config['TRAIN_DATA_DIR']
        self.checkpoints = config['MODEL_DIR']
        self.im_size = config['IMAGE_SIZE']
        self.epochs = config['EPOCHS']
        self.batch_size = config['BATCH_SIZE']
        self.classNumber = config['CLASS_NUM']
        self.lr = config['LEARNING_RATE']
        self.data_augmentation = config['IS_AUGMENTATION']
        self.rat = config['RAT']

    def get_file(self, path):
        ends = os.listdir(path)[0].split('.')[-1]
        img_list = glob.glob(os.path.join(path, '*.' + ends))
        return img_list

    def load_data(self):
        categories = list(map(self.get_file, list(map(lambda x: os.path.join(self.train_images_dir, x), os.listdir(self.train_images_dir)))))
        data_list = list(itertools.chain.from_iterable(categories))
        shuffle(data_list)
        images_data, labels_idx, labels = [], [], []

        with_platform = os.name

        label = ''
        for file in tqdm.tqdm(data_list[:7000]):
            img = cv2.imread(file)
            img = cv2.resize(src=img, dsize=(self.im_size, self.im_size), interpolation=cv2.INTER_LINEAR)
            if with_platform == 'posix':
                label = file.split('/')[-2]
            elif with_platform == 'nt':
                label = file.split('\\')[-2]
            images_data.append(img)
            labels.append(label)

        images_data = np.array(images_data, dtype='float32') / 255.0

        with open('class_idx.txt', 'r') as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
            for label in labels:
                idx = lines.index(label.rstrip())
                labels_idx.append(idx)

        labels = to_categorical(np.array(labels_idx), num_classes=self.classNumber)
        x_train, x_test, y_train, y_test = train_test_split(images_data, labels, test_size=0.1)  # 划分数据集为训练集和测试集 比例为9：1
        return x_train, x_test, y_train, y_test

    def train(self):
        logs_dir = './logs'
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(config['MODEL_DIR'], exist_ok=True)
        tensorboard = TensorBoard(log_dir=logs_dir)

        lr_reduce = keras.callbacks.ReduceLROnPlateau(monitor=config['MONITOR'],  # 当评价指标不在提升时，减少学习率
                                                      factor=0.1,  # 学习率将以lr = lr*factor的形式被减少
                                                      patience=config['LR_REDUCE_PATIENCE'],  # 需要降低学习率的训练步长
                                                      verbose=1,
                                                      mode='auto',
                                                      cooldown=0)
        early_stop = keras.callbacks.EarlyStopping(monitor=config['MONITOR'],  # （需要监视的量）当监测值相比较上一个epoch不再改善时，该回调函数将中止训练
                                                   min_delta=0,
                                                   patience=config['EARLY_STOP_PATIENCE'],  # 提前终止训练的步长（10）
                                                   verbose=1,
                                                   mode='auto')
        checkpoint = keras.callbacks.ModelCheckpoint(filepath=os.path.join(config['MODEL_DIR'], 'weights.{epoch:02d}-{loss:.2f}.h5'),
                                                     monitor=config['MONITOR'],
                                                     verbose=1,
                                                     save_best_only=True,
                                                     save_weights_only=True,
                                                     mode='auto',
                                                     period=1)

        x_train, x_test, y_train, y_test = self.load_data()

        model = keras.applications.ResNet50(include_top=True,
                                            weights=None,
                                            input_tensor=None,
                                            input_shape=(self.im_size, self.im_size, 3),
                                            pooling='max',
                                            classes=self.classNumber)
        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(lr=self.lr),
                      metrics=["accuracy"])  # compile之后才会更新权重和模型

        if self.data_augmentation:
            print("using data augmentation method")
            data_aug = ImageDataGenerator(rotation_range=5,     # 图像旋转的角度
                                          width_shift_range=0.2,    # 左右平移的比例
                                          height_shift_range=0.2,   # 上下平移参数 图片高度的某个比例
                                          zoom_range=0.3,   # 随机放大或者缩小（1-0.3~1+0.3）
                                          horizontal_flip=False,    # 随机水平翻转
                                          )
            data_aug.fit(x_train)
            model.fit_generator(generator=data_aug.flow(x_train, y_train, batch_size=self.batch_size),
                                steps_per_epoch=x_train.shape[0] // self.batch_size,
                                validation_data=(x_test, y_test),
                                shuffle=True,
                                epochs=self.epochs, verbose=1, max_queue_size=1000,
                                callbacks=[early_stop, checkpoint, lr_reduce, tensorboard],
                                )
        else:
            model.fit(x=x_train, y=y_train,
                      batch_size=self.batch_size,
                      validation_data=(x_test, y_test),
                      epochs=self.epochs,
                      callbacks=[early_stop, checkpoint, lr_reduce, tensorboard],
                      shuffle=True,
                      verbose=1)


if __name__ == '__main__':
    config_path = './config.yaml'
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)

    model = Build_Model(config)
    model.train()  # 开始训练模型
