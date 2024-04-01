# 当代人工智能实验四--文本摘要
## 10215501435 杨茜雅
本次实验是文本摘要在医学领域的应用：从详细的病情描述中提取关键信息生成简短清晰的诊断报告。

## 依赖
环境依赖已经列在 requirements.txt 中
可以依次安装，也可以
```shell
pip install -r requirements.txt
```
注意：如果使用 GPU，请安装对应版本的 PyTorch 和 CUDA。

## 结构
```
+--data
|      +--dataloader.py # 把数据使用 DataLoader 定义 batch_size, num_workers 等等训练参数加载训练
|      +--my_dataset.py # 定义如何处理数据使其符合 batch 以及 bos, eos 等等各种 token
|      +--process_data.py # 把数据从 csv 转换为列表的文件
|      +--test.csv #测试数据集from老师
|      +--train.csv #训练数据集from老师
+--model # 存放预训练后模型参数的目录
|      +--model
|      |      +--facebook
|      |      |      +--bart-base-4.pth
|      |      |      +--bart-base-9.pth #太多了就放两个，实际上用到的是-4版本
|      +--small_bart.py # 定义从 Huggingface 库选择哪个模型进行预训练
+--utils
|      +--build_lr_scheduler.py # 用于创建和配置学习率调度器（scheduler）
|      +--build_optimizer.py # 根据给定参数构建和返回一个PyTorch优化器
|      +--save_model.py # 使用 torch.save 保存模型，并且定义保存的路径
+--opts.py # 各种超参数的配置文件
+--main.py # 运行，包含训练和测试两大步骤
+--predict.py # 运行这个代码，生成预测文件
+--report.csv # 运行predict.py后会出现的预测文件，不过实验没做要求
+--requirements.txt #环境依赖
+--worddict.py # 用于处理与BERT和GPT-2语言模型相关的数据的脚本
+--10215501435 杨茜雅 当代人工智能实验四实验报告.pdf # 实验报告
```

## 运行代码

运行训练和验证步骤：进入文件根目录并且运行

```
python main.py
```

（运行预测步骤：进入文件根目录并且运行(未作要求，给出的test.csv中的diagnosis字段也不为空，可是按理来说需要预测的文件不该含有diagnosis字段)

```
python predict.py
```
会在文件根目录下生成 report.csv作为预测的输出）


对于训练和验证步骤，可以指定超参数 --num_epochs, --lr, --batch_size 等等

如果单独运行训练或者验证步骤，需要注释掉相关代码

具体在 main.py 的 59-68 行

```python
    for epoch in range(args.epochs):
        cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print('lr:', cur_lr)
        print('weight decay:', optimizer.state_dict()['param_groups'][0]['weight_decay'])
        train_loss = train_one_epoch(train_loader, model, optimizer, epoch, device=device, is_adversial=False, scaler=scaler) # 如果只运行验证，注释这两行
        valid_loss = valid(valid_loader, model, device, epoch, args.num_beams, args.file_valid) # 如果只运行训练，注释这两行

        tb_writer.add_scalar('train loss', train_loss, epoch) # 如果只运行验证，注释这两行
        tb_writer.add_scalar('valid loss', valid_loss, epoch) # 如果只运行训练，注释这两行
        tb_writer.add_scalar('lr', cur_lr, epoch)
```

