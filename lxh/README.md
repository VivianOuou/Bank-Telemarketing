## 项目概述

本项目使用UCI银行营销数据集构建和比较不同的机器学习模型，旨在预测客户是否会订购银行的定期存款产品。数据集包含多种客户属性和营销活动信息，目标变量是二分类问题（是否订购定期存款）。

好的，这是您可以用于 README 文件中描述如何通过 `ucimlrepo` 库获取数据集的中文段落：

------

### 数据集获取

本项目使用的数据集是来自 UCI 机器学习库 (UCI Machine Learning Repository) 的“银行营销 (Bank Marketing)”数据集。为了方便获取和加载数据，我们使用了 `ucimlrepo` 这个 Python 库。

可以通过以下 pip 命令安装该库：

```
pip install ucimlrepo
```

## 文件说明

项目包含四个主要的Jupyter notebook文件：

1. **EDA.ipynb** - 对原始数据的探索性分析
2. **baseline.ipynb** - 基础模型实现
3. **basline_log_processing.ipynb** - 使用对数转换的数据处理方法
4. **basline_log_smote.ipynb** - 结合对数转换和SMOTE过采样技术

## 数据处理

三个文件都实现了以下数据处理流程：

- 处理缺失值（使用"unknown"填充类别特征中的缺失值）
- 特征工程：
  - 将`pdays`（上次联系间隔天数）转换为二分类特征`pdays_was_contacted`和对数转换`pdays_log1p`
  - 将`balance`（账户余额）转换为符号类别特征`balance_sign_category`和对数转换特征`balance_log_abs_p1`

## 模型实现

### 共同点

三个文件都实现了以下模型：

- 逻辑回归 (Logistic Regression)
- 决策树 (Decision Tree)
- 随机森林 (Random Forest)
- 支持向量机 (SVM with RBF Kernel)
- 朴素贝叶斯 (Naive Bayes)
- LightGBM
- CatBoost
- 神经网络 (PyTorch简单MLP实现)

### 主要区别

1. **baseline.ipynb**：
   - 基础实现，无特殊处理

2. **basline_log_processing.ipynb**：
   - 对数值特征进行对数变换，处理长尾分布问题
   - 扩展特征工程

3. **basline_log_smote.ipynb**：
   - 在对数变换基础上增加了SMOTE过采样技术
   - 解决数据类别不平衡问题（目标变量中"no"类别占比约88%）
   - CatBoost模型参数更为复杂，针对不平衡数据进行了调整

## 模型评估

所有模型使用以下指标进行评估：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1值 (F1-score)
- ROC曲线下面积 (ROC AUC)
- G-Mean
- AUPRC

## 神经网络实现

三个文件都使用PyTorch实现了相同结构的简单多层感知机(MLP)：
- 输入层 -> 128个神经元的隐藏层 -> 64个神经元的隐藏层 -> 输出层
- 使用ReLU激活函数和Dropout(0.3)防止过拟合
- 使用BCEWithLogitsLoss损失函数
- 使用Adam优化器和余弦退火学习率调度

在所有实现中，模型训练30个epoch，并保存F1值最高的模型。



## 使用方法

要运行这些笔记本，您需要安装以下依赖后按顺序依次运行单元格：
- Python 3.x
- PyTorch
- scikit-learn
- imbalanced-learn (SMOTE)
- pandas, numpy
- matplotlib, seaborn
- lightgbm, catboost
- ucimlrepo (用于获取UCI机器学习仓库数据)

## 结论

通过对比三个实现，可以看出数据处理技术（对数转换）和处理不平衡数据的方法（SMOTE）显著改善了模型性能，特别是在少数类预测方面。

CatBoost和神经网络模型在处理这类结构化不平衡数据时表现最佳，尤其是结合了适当的数据预处理技术后。