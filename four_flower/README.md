# four-flower

![gui-test](./gui-test.png)
这是一个图像识别项目，基于 tensorflow，现有的 CNN 网络可以识别四种花的种类。适合新手对使用 tensorflow 进行一个完整的图像识别过程有一个大致轮廓。项目包括对数据集的处理，从硬盘读取数据，CNN 网络的定义，训练过程，还实现了一个 GUI 界面用于使用训练好的网络。

## Project status

这个仓库现在包含两套实现：

1. `TensorFlow 1.x + 自定义CNN`
   - 原始课程项目实现，保留为基线方案。
   - 入口文件：`train.py`、`test.py`、`gui.py`

2. `PyTorch + EfficientNet-B0`
   - 用于课程论文升级版的迁移学习方案。
   - 入口文件：`train_efficientnet.py`、`predict_efficientnet.py`
   - 主要改进：预训练模型、固定随机种子、训练/验证/测试划分、混淆矩阵、分类报告

## Require

1. 安装 Anaconda
2. 导入环境 environment.yaml  
   `conda env update -f=environment.yaml`

## Quick start

- git clone 这个项目
- 解压 input_data.rar 到你喜欢的目录。
- 修改 train.py 中

```
train_dir = 'D:/ML/flower/input_data'  # 训练样本的读入路径
logs_train_dir = 'D:/ML/flower/save'  # logs存储路径
```

为你本机的目录。

- 运行 train.py 开始训练。
- 训练完成后，修改 test.py 中的`logs_train_dir = 'D:/ML/flower/save/'`为你的目录。
- 运行 test.py 或者 gui.py 查看结果。

## EfficientNet-B0 quick start

1. 安装 PyTorch 版本依赖：

```bash
pip install -r requirements_pytorch.txt
```

2. 解压 `input_data.rar`，确保目录结构如下：

```text
input_data/
  dandelion/
  roses/
  sunflowers/
  tulips/
```

3. 运行训练：

```bash
python train_efficientnet.py --data-dir /path/to/input_data --output-dir runs/efficientnet_b0 --epochs 10
```

4. 训练完成后，输出目录会生成：
   - `best_model.pt`
   - `history.json`
   - `training_curve.png`
   - `confusion_matrix.png`
   - `classification_report.txt`

5. 运行单张图片预测：

```bash
python predict_efficientnet.py --checkpoint runs/efficientnet_b0/best_model.pt --image /path/to/test.jpg
```

## Suggested paper title

如果论文基于当前升级版实现，建议题目改为：

`基于 EfficientNet-B0 迁移学习的四类花卉图像识别系统设计与实现`
