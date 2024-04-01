# 当代人工智能实验五--多模态情感分析
## 10215501435 杨茜雅
本次实验的任务是给定配对的文本和图像，预测对应的情感标签。是一个三分类任务：positive, neutral, negative。


## 依赖
环境依赖已经列在 requirements.txt 中
可以依次安装，也可以
```shell
pip install -r requirements.txt
```
注意：如果使用 GPU，请安装对应版本的 PyTorch 和 CUDA，或者使用云主机。


## 文件夹结构
```
+--data 	老师给的数据，里面存放单个的文本和图像数据
|      +...txt
|      +...jpg
+--processed_data
|      +--valid_legacy.tsv    处理后的原始验证集数据
|      +--valid.tsv    补过标签并且翻译的验证集数据
|      +--train.tsv    补过标签并且翻译的训练集数据
|      +--test_without_label.txt   需要预测数据的 guid 和对应的空情感标签
|      +--test.tsv       处理后的测试集数据
|      +--train_legacy.tsv    处理后的原始训练集数据
|      +--train.txt   数据的 guid 和对应的情感标签
+--processors
|      +--util.py     处理如何对数据进行训练和验证
+--models
|      +--model.py    定义了 bert model 的各个部分
|      +--resnet.py    定义了 resnet 如何进行图像转换
+--pre_trained_model
|      +--resnet152.pth  预训练模型，需要从pytorch下载，下载地址在后文附上
+--predict
|      +--test_without_label_legacy.txt   利用原始数据raw data生成的预测数据
|      +--test_without_label.txt    利用增补过标签并且翻译的数据生成的预测数据(对老师给的原始数据做了neutral复制的改动)
+--output （运行sh run.sh之后会出现）
|      +--pytorch_model.bin    保存的最好模型
|      +--pytorch_encoder.bin    保存的最好模型
+--metrics
|      +--compute.py    计算加权平均F1分数，衡量分类性能
+--requirements.txt  环境依赖
+--README.md
+--results.txt  最终选择的预测输出结果
+--run.sh    任务运行脚本，运行文本 + 图像
+--run_only_image.sh   任务运行脚本，只运行图像
+--run_only_text.sh  任务运行脚本，只运行文本
+--run_test.sh   任务运行脚本，生成预测文件
+--run.py      主运行程序
+--file_tree.py   生成文件树
+--test_overfit.py       查看各个种类 label 数据的数量
+--generate_data.py     数据处理，从 txt 生成 tsv文件
+--fix_overfit.py      对于较少的标签进行增补
```


## 运行代码
需要进入文件根目录并且运行

运行模型需要先下载预训练模型 ResNet-152，并放入 pre_trained_model 文件下

链接 (https://download.pytorch.org/models/resnet152-b121ed2d.pth)

以下命令默认对于未增补过标签并且翻译的数据，输入模型并且运行。

```
sh run.sh #文本+图像
```

```
sh run_text_only.sh #仅文本
```

```
sh run_image_only.sh #仅图像
```

```
sh run_test.sh #输出预测结果
```

如果需要运行增补过标签并且翻译的数据，需要修改以上四个文件：

去除文件末尾的 `--train_file train_legacy.tsv --valid_file valid_legacy.tsv` 参数。



## Reference
[1] https://github.com/RecklessRonan/GloGNN/blob/master/readme.md

[2] https://github.com/Conyrol/CLUE_model

[3] Soleymani M, Garcia D, Jou B, et al. A survey of multimodal sentiment analysis[J]. Image and Vision Computing, 2017, 65: 3-14.

[4] Gandhi A, Adhvaryu K, Poria S, et al. Multimodal sentiment analysis: A systematic review of history, datasets, multimodal fusion methods, applications, challenges and future directions[J]. Information Fusion, 2023, 91: 424-444.

