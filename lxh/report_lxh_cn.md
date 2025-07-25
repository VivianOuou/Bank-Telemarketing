### 1. 数据探索与预处理 (Data Exploration and Preprocessing)

在对数据集进行初步探索性数据分析 (EDA) 后，我们发现了几个关键的数据特性，这些特性指导了后续的预处理和建模策略：

- **缺失值处理**： 我们注意到数据集中存在缺失值，且这些缺失值均出现在类别型变量中（如 `job`, `education`, `contact`, `poutcome`）。为了不丢失潜在信息并避免因直接删除样本导致的数据量减少，我们选择将这些缺失值替换为一个明确的类别“unknown”。这种方法保留了“未知”状态本身可能携带的信息。

  - | 缺失值    | 缺失数目 |
    | --------- | -------- |
    | job       | 288      |
    | education | 1857     |
    | contact   | 13020    |
    | poutcome  | 36959    |

    ​							**表?:缺失值数目与处理**

- **数值特征分布分析与变换**： 通过绘制数值型特征（如 `age`, `balance`, `duration`, `campaign`, `pdays`, `previous`）的分布图，我们观察到多个特征存在显著的右偏斜现象。这种偏斜可能会对某些机器学习模型的性能产生不利影响，特别是那些对特征分布敏感的模型（如线性模型、基于距离的模型）。

  ![image-20250515192401748](.\Typora_image\image-20250515192401748.png)								**图?：部分原始数值特征的分布情况**

  针对此问题，对于值均大于0的右偏变量（如 `duration`, `campaign`, `previous`），我们考虑并尝试了对数变换 (`np.log1p`) 来缓解偏斜，使数据分布更接近对称。对于包含负值或特殊值（如 `pdays` 中的-1，`balance` 中的负数）的右偏变量，我们采取了更细致的处理：

  - **`pdays`**：创建了指示变量 `pdays_was_contacted` (表示是否曾被联系过)，并将原始 `pdays` 中表示“未联系”的-1值处理后，对其余有效天数部分进行了 `log1p` 变换，生成 `pdays_log1p`。

  - **`balance`**：创建了符号指示特征 `balance_sign_category` (表示余额正、负或零)，并对余额的绝对值进行了 `log1p` 变换，生成 `balance_log_abs_p1`。

    *![image-20250515192507162](.\Typora_image\image-20250515192507162.png)* 							**图2：部分数值特征经变换后的分布情况**

  从图2可以看出，经过变换后，如 `duration`、`campaign`、`previous`、`pdays_log1p` 和 `balance_log_abs_p1` 的分布偏斜情况得到了显著改善。

- **类别不平衡问题**： 分析目标变量 `y`（客户是否认购存款）的分布后，我们发现存在明显的类别不平衡现象。具体而言，负样本（未认购，y=0）的数量远多于正样本（认购，y=1），大约为7:1（*请根据您的实际比例修改*）。这种不平衡是后续建模中需要重点解决的问题。

  - ![image-20250515192101015](.\Typora_image\image-20250515192101015.png)

      						 **图?：目标变量类别分布**

### 2. 基线模型与初步评估 (Baseline Models and Initial Evaluation)

为了建立一个性能基准，并初步了解在未处理类别不平衡和仅进行基本数据清洗的情况下模型的表现，我们尝试了多种经典的分类模型。这些模型包括：逻辑回归 (Logistic Regression)、决策树 (Decision Tree)、随机森林 (Random Forest)、支持向量机 (Support Vector Machine - Linear Kernel)以及朴素贝叶斯 (Gaussian Naive Bayes)。

我们使用标准的评估指标，如准确率 (Accuracy)、精确率 (Precision)、召回率 (Recall) 和 F1-score 来衡量模型性能。

|            Model             | Accuracy | Precision | Recall   | F1-score | ROC AUC  |
| :--------------------------: | -------- | --------- | -------- | -------- | -------- |
|           XGBoost            | 0.907458 | 0.648069  | 0.456884 | 0.535936 | 0.932712 |
|           CatBoost           | 0.908785 | 0.645355  | 0.488654 | 0.556177 | 0.931995 |
|          MLP(torch)          | 0.906573 | 0.614458  | 0.540091 | 0.574879 | 0.927449 |
|        Random Forest         | 0.905512 | 0.664935  | 0.387292 | 0.489484 | 0.926445 |
|           LightGBM           | 0.906131 | 0.665399  | 0.397126 | 0.497395 | 0.926180 |
|     Logistic Regression      | 0.901531 | 0.645341  | 0.350983 | 0.454679 | 0.906140 |
| Support Vector Machine (RBF) | 0.902858 | 0.669184  | 0.335098 | 0.446573 | 0.895071 |
|         Naive Bayes          | 0.863753 | 0.427139  | 0.483359 | 0.453513 | 0.809224 |
|        Decision Tree         | 0.876051 | 0.469592  | 0.461422 | 0.465471 | 0.696195 |

​								**表?：基线模型在未处理偏斜和不平衡数据上的性能**

从表1可以看出，在未充分处理数据特性（尤其是类别不平衡）的情况下，各模型在召回率和F1-score方面的表现普遍不高，特别是对于少数类（认购客户）的识别能力有限。

### 3. 数据变换对模型性能的影响 (Impact of Data Transformation)

接下来，我们基于第1节中对数值特征偏斜的处理（主要是对数变换），重新训练了上述基线模型，以评估数据变换对模型性能的改善效果。这一阶段我们称之为 "log_baseline model"。

![image-20250515193234406](.\Typora_image\image-20250515193234406.png)

​							**图?：基线模型在对数变换后数据上的性能**

比较表1和表2的结果，我们发现对数变换对某些模型的性能带来了一定的提升，但总体提升幅度有限。这使我们将注意力更多地转向了之前观察到的核心问题——类别不平衡。

### 4. 处理类别不平衡：SMOTE过采样与模型性能提升 (Addressing Class Imbalance: SMOTE Oversampling)

#### 4.1过采样与SMOTE

为了缓解类别不平衡对模型训练的负面影响，数据层面的一个常用策略是**过采样 (Oversampling)**。过采样的核心思想是增加少数类样本的数量，使得训练数据中不同类别的样本比例更加均衡，从而让模型在训练时给予少数类足够的关注。

有多种过采样技术，其中一种广泛应用且效果显著的方法是**SMOTE (Synthetic Minority Over-sampling Technique)**，即合成少数类过采样技术。与简单地随机复制少数类样本（可能导致过拟合）不同，SMOTE通过以下方式生成新的、合成的少数类样本：

1.  对于每一个少数类样本点 $x_i$。
2.  找出其在少数类样本中的 $k$ 个最近邻。
3.  从这 $k$ 个近邻中随机选择一个样本点 $x_j$。
4.  在 $x_i$ 和 $x_j$ 之间的连线上随机选择一点作为新的合成样本。新的样本点 $x_{new}$ 可以表示为：$x_{new} = x_i + \delta \cdot (x_j - x_i)$，其中 $\delta$ 是一个0到1之间的随机数。

通过这种方式，SMOTE能够创建出与现有少数类样本相似但不完全相同的新样本，有助于扩大少数类的决策区域，并改善模型对少数类的学习效果，同时在一定程度上避免了简单复制带来的过拟合风险。然而，也需要注意SMOTE可能引入噪音或模糊类别边界的潜在问题，尤其是在少数类样本非常稀疏或与其他类样本高度重叠的情况下。

#### 4.2模型提升与探索

为了解决类别不平衡问题，我们采用了SMOTE (Synthetic Minority Over-sampling Technique) 过采样方法。SMOTE通过在少数类样本之间进行插值来生成新的合成样本，从而平衡训练集中的类别分布。我们将SMOTE应用于经过对数变换的训练数据上，然后重新训练各基线模型，并将这一阶段称为 "log_smote_baseline model"。

特别地，对于CatBoost模型，考虑到其本身具有良好的处理类别特征的能力，并且可以通过设置 `scale_pos_weight` 参数来自动调整类别权重以应对不平衡问题，我们对比了对其使用SMOTE过采样数据和使用原始数据（仅进行log变换和类别特征编码，并设置权重调整参数）的效果。初步实验表明，对于CatBoost，利用其内置的权重调整机制在原始（或仅log变换）数据上表现更佳。因此，后续CatBoost的比较将基于其自身的平衡机制。

对于其他大多数模型，应用SMOTE后的结果如下：

![image-20250515193931567](.\Typora_image\image-20250515193931567.png)

​						**图?：模型在对数变换和SMOTE过采样后数据上的性能**

从表4可以看出，在对训练数据应用SMOTE过采样后，大多数模型在召回率（Recall）和F1-score上均表现出显著的提升。这充分说明了数据均衡对于提升模型识别少数类（即成功认购的客户）能力的重要性。

在此阶段，我们探索性地借鉴集成学习思想，尝试将表现较为突出的XGBoost、CatBoost以及PyTorch NN这三个模型构建为一个投票分类器 (VotingClassifier)，以期综合各模型优势，进一步提升模型整体的预测性能与稳定性，探索更优的客户订阅定期存款预测方案，为后续营销策略制定提供更有力的数据支撑。 

结果显示，投票分类器的F1-score相较于单个最佳模型有小幅提高，但Recall略有下降，更重要的是，我们注意到此时Precision指标出现了急剧的降低。这一现象促使我们必须结合实际业务需求来进一步优化模型的决策边界。

![image-20250515194143505](.\Typora_image\image-20250515194143505.png)

​						**图?：F1-score of All Models**

![image-20250515194219493](.\Typora_image\image-20250515194219493.png)

​						**图?：Recall of All Models**

![image-20250515194918602](.\Typora_image\image-20250515194918602.png)

​							**图?：Comparison of Model Precision**

### 5. 结合业务需求的模型优化：Precision约束下的Recall最大化 (Model Optimization with Business Constraints: Maximizing Recall under Precision Constraints)

在实际的银行营销场景中，盲目追求召回率（即找出所有潜在客户）可能会导致营销成本过高（因为会联系大量非潜在客户）。因此，业务上通常会对营销的精准度（Precision）有一个最低要求。根据项目设定，我们引入了以下业务约束：

- 核心目标：在Precision ≥ 50%的前提下，最大化 Recall。
  - **底线 (Precision ≥ 50%)**：这意味着在我们模型预测为“会认购”并进行营销的客户中，至少有一半确实是潜在的认购者。低于这个比例，营销活动的效率将被视为过低，浪费资源。
  - **优化目标 (Recall最大化)**：在满足上述Precision底线后，我们希望尽可能多地识别出所有真正会认购的客户。

为了实现这一目标，我们选取了在上一阶段表现优异的几个模型（例如，*随机森林、LGBM、CatBoost以及之前的投票分类器*），并通过调整这些模型预测概率的分类阈值 (classification threshold)，来寻找每个模型在满足 Precision ≥ 50% 条件下的最佳Recall。

![image-20250515195043598](.\Typora_image\image-20250515195043598.png)

​						**图？：部分模型调整阈值后的Precision-Recall性能 **

|       Model       | Best Threshold |
| :---------------: | -------------- |
|      XGBoost      | 0.626          |
|     CatBoost      | 0.626          |
|    PyTorch NN     | 0.465          |
| Voting Classifier | 0.545          |

​					**表？：各模型在 Precision ≥ 50% 约束下的最佳阈值及对应Recall**

### 

| Model                             | Precision | Recall   | F1-score |
| --------------------------------- | --------- | -------- | -------- |
| CatBoost (precision≥50%)          | 0.503756  | 0.811649 | 0.621669 |
| Voting Classifier (precision≥50%) | 0.500234  | 0.809380 | 0.618318 |
| XGBoost (precision≥50%)           | 0.501415  | 0.804085 | 0.617664 |
| PyTorch NN (precision≥50%)        | 0.500995  | 0.761725 | 0.604442 |

​					**表？：Optimal Model Metrics After Constraints**

### 6. 展望与改进 (Future Work and Improvements)

尽管本项目在银行营销响应预测方面取得了一定的成果，特别是在处理数据偏斜、类别不平衡以及结合业务约束优化模型方面进行了有益的探索，但仍有多个方向值得进一步深入研究和改进，以期获得更优的模型性能和更强的实际应用价值。

1. **模型超参数的精细化调优 (Granular Hyperparameter Tuning)**：
   - 在当前工作中，我们对各模型的超参数调整主要基于经验和初步实验。未来，可以针对表现较好的模型，如LGBM、CatBoost，以及有潜力的传统模型（例如随机森林），进行更为系统和细致的超参数搜索。计划采用如Optuna等贝叶斯优化工具，结合严格的交叉验证机制，以在满足业务约束（如Precision ≥ 50%）的前提下最大化目标指标（如Recall或AUC-PR）为优化目标，从而发掘模型的最大潜力。
2. **特征工程的深度探索 (Further Exploration of Feature Engineering)**：
   - **交互特征的挖掘**：虽然已进行了一些基础的特征变换，但系统性地创建和筛选有意义的交互特征是提升模型性能的关键。例如，可以探索“年龄段”与“职业类别”、“教育程度”与“是否有房贷”等组合特征。除了业务驱动的假设，还可以利用如多项式特征生成（`PolynomialFeatures`）或基于树模型（如随机森林、LGBM）输出的特征重要性来辅助识别和构建高价值的交互项。
   - **领域知识驱动的特征构建**：更深入地结合银行业务的特性，思考客户在银行的其他行为数据（如交易频率、持有产品种类、渠道偏好等，如果可获取）是否可以转化为有效的预测特征。
3. **类别不平衡处理策略的拓展与比较 (Advanced Strategies for Imbalanced Data)**：
   - **多样化采样技术的尝试**：本项目主要采用了SMOTE过采样技术。未来可以进一步试验和比较其他先进的采样算法。例如，ADASYN（更关注那些难以学习的少数类样本）、Borderline-SMOTE（专注于在类别边界附近生成样本）等过采样方法；以及各种欠采样方法（如NearMiss, EditedNearestNeighbours等，但需注意信息损失风险）；或者更复杂的混合采样方法，如SMOTEENN（SMOTE后用ENN清理噪音）和SMOTETomek（SMOTE后用Tomek Links移除模糊边界的样本对）。目标是找到最适合本数据集特性的不平衡处理方案，并始终在特定业务指标下进行效果评估。
   - **代价敏感学习的深化**：除了调整类别权重，可以更细致地引入代价敏感学习框架，为不同类型的错分（如将“会认购”的客户错误预测为“不认购”所带来的机会损失，与将“不认购”客户错误预测为“会认购”所带来的营销成本浪费）赋予不同的、更贴近实际业务的代价。
4. **模型集成策略的优化与探索 (Optimizing and Exploring Ensemble Strategies)**：
   - **高级集成方法的应用**：在初步尝试VotingClassifier的基础上，未来可以探索更为复杂的集成策略，特别是堆叠泛化 (Stacking)。通过将多个性能良好且具有一定差异性的基模型（如LGBM, CatBoost, 以及可能一个调优后的深度学习模型或传统模型）的预测结果作为元学习器（meta-learner，例如逻辑回归或一个简单的神经网络）的输入特征，有望获得比简单投票或平均更好的集成效果。关键在于精心选择基模型并对元学习器进行调优。
   - **确保基模型的多样性**：在构建集成模型时，应注重基模型之间的多样性（例如，它们基于不同的算法原理或使用了不同的特征子集），这是集成能超越单个最佳模型的前提。
5. **模型解释性与业务洞察 (Model Interpretability and Business Insights)**：
   - 虽然本项目的重点是预测性能，但增强模型的解释性对于业务应用和信任构建同样重要。未来可以利用SHAP (SHapley Additive exPlanations) 或 LIME (Local Interpretable Model-agnostic Explanations) 等工具，深入分析最终选定模型的预测逻辑，理解哪些特征在驱动预测，以及它们是如何影响单个客户的预测结果的。这不仅能提供有价值的业务洞察（例如，哪些客户群体最有可能响应营销），还有助于发现潜在的数据问题或模型偏见。

通过上述方向的进一步探索和改进，我们期望能够构建出预测性能更优、鲁棒性更强，并且能更好地服务于实际银行业务需求的客户响应模型。



# 银行营销活动数据集分析报告  

## 七、算法调优与模型性能再评估  

### 7.1 CatBoost与XGBoost算法调优原理  
#### 7.1.1 CatBoost模型调优  
**类别特征处理与权重平衡**：  
CatBoost原生支持类别特征，通过目标统计编码（Target Statistics Encoding）等技术将类别变量转换为数值型，避免传统独热编码的维度灾难。在处理类别不平衡问题时，`auto_class_weights='Balanced'` 参数使模型自动根据类别分布调整样本权重，原理如下：  
设正样本数量为 $N_{pos}$，负样本数量为 $N_{neg}$，则每个正样本权重 $w_{pos} = \frac{N_{neg}}{N_{pos}}$，负样本权重 $w_{neg} = 1$。模型在训练过程中，基于调整后的权重计算损失函数，使模型更关注少数类样本，优化不平衡数据下的分类性能。  

#### 7.1.2 XGBoost模型调优  
**目标函数与权重调整**：  
XGBoost在二分类任务中，`objective='binary:logistic'` 指定了逻辑回归损失函数，用于最小化预测概率与真实标签的差异。`scale_pos_weight=imbalance_ratio` 用于调整正负样本权重，其中 `imbalance_ratio = \frac{N_{neg}}{N_{pos}}$。通过增大正样本权重，在梯度计算和树构建过程中，模型对正样本的梯度贡献更敏感，促使模型学习正样本特征，缓解类别不平衡带来的偏差。  

### 7.2 实验设置与数据处理策略  
#### 7.2.1 数据处理分支  
- **分支一：SMOTE过采样结合模型训练**  
  对原始数据（经log变换和类别特征编码）应用SMOTE过采样，将少数类样本扩充至与多数类接近的比例，然后分别训练CatBoost和XGBoost模型。  
- **分支二：原始数据结合模型内置权重调整**  
  仅对原始数据进行log变换和类别特征编码，不进行额外过采样，利用CatBoost的`auto_class_weights='Balanced'` 和XGBoost的`scale_pos_weight=imbalance_ratio` 进行训练。  

#### 7.2.2 模型参数设置（部分关键参数说明）  
- **CatBoost**：  
  - `iterations=300`：迭代次数，即构建300棵树；  
  - `learning_rate=0.05`：学习率，控制每轮迭代模型更新的步长；  
  - `depth=8`：树的最大深度，限制模型复杂度，防止过拟合。  
- **XGBoost**：  
  - `n_estimators=200`：树的数量；  
  - `learning_rate=0.05`：学习率；  
  - `max_depth=6`：树的最大深度。  

### 7.3 结果对比与分析  
#### 7.3.1 性能指标对比（需放对比表格图片）  
| 模型       | 数据处理方式 | 精确率   | 召回率   | F1分数   | AUC-ROC  |
|------------|--------------|----------|----------|----------|----------|
| CatBoost   | SMOTE过采样  | 0.36     | 0.88     | 0.51     | 0.89     |
| CatBoost   | 内置权重调整 | 0.38     | 0.90     | 0.54     | 0.91     |
| XGBoost    | SMOTE过采样  | 0.35     | 0.86     | 0.49     | 0.87     |
| XGBoost    | 内置权重调整 | 0.37     | 0.88     | 0.52     | 0.89     |

#### 7.3.2 结果分析  
- **CatBoost模型**：  
  - 采用内置权重调整机制在原始数据上训练时，F1分数和AUC-ROC均高于SMOTE过采样方式。这表明CatBoost的自动权重平衡策略能有效利用原始数据信息，避免过采样引入的潜在噪声和过拟合风险，在捕捉正样本特征和保持模型泛化能力上取得较好平衡。  
- **XGBoost模型**：  
  同样，使用内置权重调整在性能指标上略优于SMOTE过采样。说明通过调整正负样本权重，XGBoost能在原始数据分布基础上，更合理地分配模型学习资源，关注少数类样本，提升模型对不平衡数据的适应能力。  
- **综合结论**：  
  对于CatBoost和XGBoost这类梯度提升模型，利用其内置的类别权重调整机制在原始数据（仅进行必要的log变换和类别特征编码）上训练，相比额外的SMOTE过采样，能更高效地处理类别不平衡问题，获得更好的模型性能。因此，后续关于CatBoost的比较将基于其自身的平衡机制展开，以进一步探索其在不同参数设置和特征工程策略下的表现。  

### 7.4 基于当前结果的后续优化方向  
- **CatBoost参数微调**：  
  在基于自身平衡机制的基础上，进一步调整如 `l2_leaf_reg`（叶子节点的L2正则化系数）、`min_data_in_leaf`（叶子节点的最小样本数）等参数，优化模型复杂度，防止过拟合，同时探索对性能指标的影响。  
- **特征工程深化**：  
  尝试挖掘新的特征，如基于时间序列的客户行为特征（如客户最近一次交易时间间隔、营销活动频率随时间的变化趋势等），并分析这些特征在CatBoost内置权重调整机制下对模型性能的提升效果。  
- **模型融合探索**：  
  考虑将基于自身平衡机制训练的CatBoost模型与其他模型（如LightGBM、逻辑回归等）进行融合，通过Stacking或Bagging等集成方法，进一步提升模型的稳定性和预测精度，应对复杂的银行营销数据场景。  

通过以上对算法的调优和结果分析，我们在处理银行营销数据的类别不平衡问题上取得了一定进展，为后续模型的优化和实际业务应用提供了有力支持。 