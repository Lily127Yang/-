# 当代人工智能实验三--图像分类及经典CNN实现
除了实现必选的三种架构: LeNet，AlexNet，ResNet之外，还实现了VGGNet,MobileNet和GoogLeNet一共六个架构。

## 依赖
环境依赖已经列在 requirements.txt 中

## 结构

```
|--10215501435-杨茜雅-当代人工智能实验三.pdf  #实验报告
|-- LeNet.py	# LeNet模型实现
|-- AlexNet.py  # AlexNet模型实现
|-- ResNet.py  # ResNet模型实现
|-- GoogLeNet.py   # GoogLeNet模型实现
|-- MobileNet.py   # MobileNet模型实现
|-- VGGNet.py   # VGGNet模型实现
|-- README.md
|-- requirements.txt   # 环境依赖
|-- images # 一些截图
    |-- 参数含义.png # 各个参数的含义
    |-- 网络结构summary.png # torchvision.summary 输出的网络结构 summary，和每个 epoch 运行时的训练集损失、训练集准确率、验证集损失、验证集准确率
|-- cnn.py # 脚本文件
|-- data # 实验数据集
|-- cnn.ipynb & cnn2.ipynb # 一些test代码
```    

## Usage

输入命令

```
python cnn.py -model LeNet --num_epochs 5 --batch_size 64 -lr 0.0001 -dropout 0.5 
python cnn.py -model AlexNet --num_epochs 5 --batch_size 64 -lr 0.0001 -dropout 0.5
python cnn.py -model ResNet --num_epochs 5 --batch_size 64 -lr 0.0001 -dropout 0.5
python cnn.py -model VGGNet --num_epochs 5 --batch_size 64 -lr 0.0001 -dropout 0.5
python cnn.py -model MobileNet --num_epochs 5 --batch_size 64 -lr 0.0001 -dropout 0.5
python cnn.py -model GoogLeNet --num_epochs 5 --batch_size 64 -lr 0.0001 -dropout 0.5
 
```

点击回车即可运行。

你可以看到 torchvision.summary 输出的网络结构 summary，和每个 epoch 运行时的训练集损失、训练集准确率、验证集损失、验证集准确率。

cuda 版本：12.1

