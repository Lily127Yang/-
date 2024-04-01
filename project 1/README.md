# 当代人工智能实验一--文本分类
利用逻辑回归/SVM/MLP/决策树/Bert模型进行十分类的英文文本分类。

## 依赖
- pandas == 1.5.3
- scikit-learn == 1.3.0
- numpy == 1.24.3
- keras == 2.13.1
- nltk == 3.8.1
- datasets == 2.14.5
- transformers == 4.34.0
- torch == 2.0.0

环境依赖已经列在 requirements.txt 中
## 安装
pip install pandas
pip install scikit-learn
pip install numpy
pip install keras
pip install nltk
pip install datasets
pip install transformers
pip install torch
或者
```shell
pip install -r requirements.txt
```
注意：如果使用 GPU，请安装对应版本的 PyTorch 和 CUDA。


## 结构

```
|--当代AI实验一实验报告-10215501435杨茜雅.pdf  #实验报告
|-- MLP.py	# MLP模型代码
|-- SVM.py  # SVM模型代码
|-- tree.py   # 决策树模型代码
|-- LR.py # 逻辑回归模型代码
|-- optimise.py #决策树模型调参代码
|-- optimise_save.ipynb #可观测到结果的调参代码以及比较不同输出的代码，不用运行！没个24h运行不完的，看结果就行
|-- results.txt # 表现最优的(MLP模型)预测结果
|-- train_data.txt	# 训练集
|-- test.txt	# 测试集
|-- train_data.csv  #训练集的csv形式
|-- test.csv #测试集的csv形式
|-- README.md
|-- results.txt  #最终决定的最好的预测结果（与Bert文件夹下的results_800.txt内容一样）  
|-- requirements.txt   #环境
|-- 运行截图1&2.png # 演示图
|--Bert  #Bert模型相关代码
    |--train_model.py           #训练模型的代码
    |--use_model.py             #用于预测的代码
    |-- results_800.txt        #第二轮训练结束的模型的预测结果
    |-- results_1600.txt       #第四轮训练结束的模型的预测结果
    |-- results_2000.txt       #第五轮训练结束的模型的预测结果
    |--best_model		# 手动创建的文件，用来存放已训练好的模型和参数，是原本output文件夹里的checkpoint-800，目前这个文件夹下的内容和checkpoint-800文件夹里的内容完全一样
        |--config.json
        |--pytorch_model.bin
        |--training_args.bin
    |--dataset												# 数据
        test.txt										# 测试数据
        train_data.txt									# 训练数据
    |--output											# 模型结果输出			
        |-- history_socre.json  #留了一个可以看到历史训练分数的文件，原先在checkpoing-2000文件夹里的，单独抽出来了
        #原本有五个训练的checkpoint文件夹的，但是太大了，最好的那个模型已经被搬到best_model文件夹里用于use_model.py了          
```



## Usage

在终端中输入

```shell
python MLP.py
python SVM.py
python TextCNN.py
python 决策树.py
python 逻辑回归.py
进入Bert文件夹
python train_model.py
```

或者
```vscode或者其他编辑器中
打开本次作业提交的文件夹
挨个运行
LR.py
SVM.PY
MLP.py
tree.py
进入Bert文件夹，运行use_model.py即可


