## 简介
本项目是基于ResNet50调查问卷手写结果实现图片分类

## 准备数据集
1. 训练集各分类存储路径为: `./dataset/train/`
2. 测试集各分类存储路径为: `./dataset/test/`
3. 训练前，可在configs/config.yaml中修改相关配置参数

## 训练数据示例
|分类名称|图像示例|分类描述
|-----|-----|-----|
|01|![](./images/01.jpg)|checkbox未勾选
|02|![](./images/02.jpg)|checkbox勾选
|03|![](./images/03.jpg)|字母或数字没有圈
|04|![](./images/04.jpg)|字母或数字有圈
|05|![](./images/05.jpg)|T
|06|![](./images/06.jpg)|F
|07|![](./images/07.jpg)|勾号
|08|![](./images/08.jpg)|叉号
|09|![](./images/09.jpg)|Y/y
|10|![](./images/10.jpg)|N/n
|11|![](./images/11.jpg)|YES/yes
|12|![](./images/12.jpg)|NO/no
|13|![](./images/13.jpg)|圆圈
|14|![](./images/14.jpg)|空白


## 模型训练
- 为训练集和测试集生成.csv文件
```Base
python gen_train_csv.py
```
- 启动训练
```Base
python train.py
```

## 模型预测
```Base
python predict.py
```
