# 在GRNavigation框架下自定义模型的教程

本文档详细说明如何在GRNavigation项目中创建自定义的模型（Model）和智能体（Agent）。 通过遵循本指南，您可以成功地在GRNavigation项目中集成自己的导航算法，并进行有效的训练和评估。

## 📋 目录

1. [概述](#概述)
2. [模型训练](#模型训练)
   - [创建自定义Model类](#创建自定义model类)
   - [创建自定义Config类](#创建自定义config类)
   - [注册和集成](#注册和集成)
   - [创建自定义Agent](#创建自定义agent)
   - [创建训练器](#创建训练器)
   - [创建数据集（如需要）](#创建数据集如需要)
   - [设置对应的配置](#设置对应的配置)
3. [模型评估](#模型评估)
   - [设置对应的配置](#设置对应的配置-1)
   - [下载GRUtopia资源](#下载grutopia资源)
4. [完整示例](#完整示例)
   - [训练和评估自定义模型](#训练和评估自定义模型)
   - [训练模型代码示例](#训练模型代码示例)
   - [评估模型代码示例](#评估模型代码示例)

## 概述

GRNavigation项目采用模块化设计，允许开发者轻松添加新的导航算法。主要组件包括：

- **Model**: 实现具体的神经网络架构和推理逻辑
- **Agent**: 作为Model的包装器，处理环境交互和数据预处理
- **Config**: 定义模型和训练的配置参数

## 模型训练

### 创建自定义Model类

Model是模型的具体实现，输入为当前的机器人的状态信息，包括摄像头，位置信息等，输出应为机器人应做的动作。根据controller的定义，输出的action应该为`List[int]`，其对应的动作如下：
- 0: stop
- 1: move forward
- 2: turn left
- 3: turn right

具体controller的实现参考[`discrete_controller.py`](../grnavigation/projects/grutopia_vln_extension/controllers/discrete_controller.py)

所有自定义模型都应该继承自`PreTrainedModel`，并实现必要的方法，具体例子可以参考[`cma_model.py`](../grnavigation/model/cma/cma_policy.py)中的`CMANet`。

### 创建自定义Config类
在Model文件中定义Config，Config应该继承自`PretrainedConfig`，具体实现可参考[`cma_model.py`](../grnavigation/model/cma/cma_policy.py)中的`CMAModelConfig`。

### 注册和集成

在[`grnavigation.model`](../grnavigation/model/__init__.py)中的`get_policy`中添加新模型，在`get_config`中添加新模型的配置

### 创建自定义Agent

Agent负责与环境的交互、数据的预处理和后处理，并调用Model进行推理。自定义Agent通常需要继承自[`Agent`](../grnavigation/agent/base.py)，并实现如下关键方法：

- `reset()`：重置Agent的内部状态（如RNN状态、历史动作等），通常在每个episode开始时调用。
- `inference(obs)`：接收环境观测`obs`，进行必要的预处理（如tokenize指令、pad等），调用模型推理，并返回动作。
- `step(obs)`：对外接口，通常调用`inference`，并可包含额外的日志或计时。

具体例子可参考[`CMAAgent`](../grnavigation/agent/cma_agent.py)

### 创建训练器

训练器负责模型的训练流程管理，包括数据加载、前向传播、损失计算、反向传播等。自定义训练器通常需要继承自[`基础训练器类`](../grnavigation/trainer/base.py)，并实现如下关键方法：

- `train_epoch()`：执行一个训练epoch，包括数据批次迭代、模型前向传播、损失计算和参数更新。
- `eval_epoch()`：执行模型评估，在验证集上测试模型性能并记录指标。
- `save_checkpoint()`：保存模型检查点，包括模型权重、优化器状态和训练进度。
- `load_checkpoint()`：加载预训练模型或恢复训练状态。

具体实现可参考[`CMATrainer`](../grnavigation/trainer/cma_trainer.py)，该训练器展示了如何处理序列数据、计算动作损失以及实现模仿学习的训练逻辑。

<!-- 训练器还需要处理以下关键功能：
- **数据加载**：配置DataLoader，处理批次数据的预处理和增强
- **损失函数**：根据任务特点选择合适的损失函数（如交叉熵、MSE等）
- **优化器配置**：设置学习率调度、权重衰减等训练超参数
- **日志记录**：记录训练损失、验证指标和tensorboard可视化
- **早停机制**：监控验证性能，避免过拟合 -->


### 创建数据集（如需要）

如果您的模型需要特殊的数据预处理，可以创建新的数据集类，数据集类需继承自[`BaseDataset`](../grnavigation/dataset/base.py)，并实现以下关键方法：

- `_load_next`：负责从数据集中加载下一个样本，返回一个观测字典（dict），包含模型所需的所有输入字段。通常需要结合数据索引、LMDB等存储方式实现数据的高效读取和解码。

具体实现可以参照[`cma_dataset.py`](../grnavigation/dataset/cma_dataset.py)

### 设置对应的配置
可参考以下现有配置文件进行自定义配置：

- **CMA模型配置**：[`cma_exp_cfg`](../scripts/train/configs/cma.py)
<!-- - **Seq2Seq模型配置**：[`seq2seq_exp_cfg`](../scripts/train/configs/seq2seq.py)  
- **RDP模型配置**：[`rdp_exp_cfg`](../scripts/train/configs/rdp.py) -->

配置文件需要定义实验配置(`ExpCfg`)、评估配置(`EvalCfg`)和训练配置(`IlCfg`)，并确保在[`__init__.py`](../scripts/train/configs/__init__.py)中正确导入和注册您的配置。

主要配置项包括：
- `name`：实验名称
- `model_name`：模型名称，需与模型注册时的名称一致
- `batch_size`：批处理大小
- `lr`：学习率
- `epochs`：训练轮数
- `dataset_*_root_dir`：数据集路径
- `lmdb_features_dir`：特征存储路径

## 模型评估
为了提升评估效率，您的自定义模型需要兼容并行评测框架。在完成Model和Agent的实现后，建议对模型进行并行评估测试，以确保其能够正确支持多环境并行推理。我们将提供10条样本数据进行并行评测。请特别注意，在调用reset方法时，模型应能将指定环境的所有相关状态变量恢复为初始默认值，确保每个环境的独立性和评测的准确性。

### 设置对应的配置
可参考以下现有评估配置文件进行自定义配置：

- **CMA模型评估配置**：[`h1_cma_cfg.py`](../scripts/eval/configs/h1_cma_cfg.py)

评估配置文件需要定义以下主要配置项：
- `name`：评估实验名称
- `model_name`：模型名称，需与训练时注册的名称一致
- `ckpt_to_load`：要加载的模型检查点路径
- `split`：评估数据集分割（如'val_seen', 'val_unseen', 'test'等）

### 下载GRUtopia资源

为确保GRUtopia框架正确运行，为保证GRUtopia正确运行，需下载[相关资源](https://xxxxxx)。在eval时将资源路径加入环境变量，具体可参考[start_eval.sh](../scripts/eval/start_eval.sh)

## 完整示例

### 训练和评估自定义模型

```bash
# 训练
./scripts/train/start_train.sh --name train_task_name --config scripts/train/configs/custom.py

# 评估
./scripts/eval/start_eval.sh --grutopia_assets_path path/to/grutopia_assets --config scripts/eval/configs/custom.py 
```

### 训练模型代码示例
训练模型的代码可见
- [`grnavigation/agent/cma_agent.py`](../grnavigation/agent/cma_agent.py)
- [`grnavigation/model/cma/cma_policy.py`](../grnavigation/model/cma/cma_policy.py)
- [`grnavigation/configs/model/cma.py`](../grnavigation/configs/model/cma.py)
- [`scripts/train/configs/cma.py`](../scripts/train/configs/cma.py)

如果需要自己定义task，目前框架所使用的task代码在
- [`grnavigation/projects/grutopia_vln_extension/tasks/vln_eval_task.py`](../grnavigation/projects/grutopia_vln_extension/tasks/vln_eval_task.py)

可供参考

### 评估模型代码示例
评估模型代码可见
- [`grnavigation/agent/cma_agent.py`](../grnavigation/agent/cma_agent.py)
- [`grnavigation/model/cma/cma_policy.py`](../grnavigation/model/cma/cma_policy.py)
- [`grnavigation/configs/model/cma.py`](../grnavigation/configs/model/cma.py)
- [`scripts/eval/configs/h1_cma_cfg.py`](../scripts/eval/configs/h1_cma_cfg.py)

可供参考

