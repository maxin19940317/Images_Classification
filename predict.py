"""
测试单张图像分类
"""

import os
import yaml
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from senet import senet


def predict_single(test_image, model):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((400, 400)),
                                    transforms.CenterCrop(384),
                                    transforms.ToTensor(),
                                    normalize,
                                    ])
    test_image_tensor = transform(test_image)

    test_image_tensor = test_image_tensor.view(1, 3, 384, 384).cuda()
    with torch.no_grad():
        model.eval()    # 固定BatchNormalization和Dropout

        pred = model(test_image_tensor)
        """
        将张量的每个元素缩放到(0, 1)区间且和为1
        dim为0，按列计算
        dim为1，按行计算
        """
        m = nn.Softmax(dim=1)

    prob_list = m(pred)[0].tolist()
    label_index = prob_list.index(max(prob_list))
    return label_index


if __name__ == '__main__':
    model = senet.senet_new_154(num_classes=14)
    model = torch.nn.DataParallel(model).cuda()

    with open('./configs/config.yaml') as f:
        config = yaml.load(f)
    best_model = torch.load(os.path.join('models', config['MODEL']['PRE_MODEL_NAME'], 'model_best.pth.tar'))
    model.load_state_dict(best_model['state_dict'])

    # 读取测试图片列表
    test_data_list = pd.read_csv('data/test.csv')
    error_num = 0
    for index, row in test_data_list.iterrows():
        img_test = Image.open(row['image_path'])
        pred_index = predict_single(test_image=img_test, model=model)
        if pred_index != row['label_index']:
            error_num = error_num + 1
            print(row['image_path'], row['label_index'], pred_index)

    print('acc:', 1 - error_num / len(test_data_list))

    # 释放GPU缓存
    torch.cuda.empty_cache()
