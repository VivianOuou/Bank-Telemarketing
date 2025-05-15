### 1. Data Exploration and Preprocessing

After conducting an initial Exploratory Data Analysis (EDA) on the dataset, we identified several key data characteristics that guided subsequent preprocessing and modeling strategies:

- **Missing Value Handling**: We noticed the presence of missing values in the dataset, all of which appeared in categorical variables (such as `job`, `education`, `contact`, `poutcome`). To avoid losing potential information and to prevent a reduction in data volume due to direct sample deletion, we chose to replace these missing values with a distinct category "unknown". This approach preserves the information that the "unknown" state itself might carry.

  - | Missing Value | Count |
    | ------------- | ----- |
    | job           | 288   |
    | education     | 1857  |
    | contact       | 13020 |
    | poutcome      | 36959 |

    **Table ?: Number of Missing Values and Handling**

- **Numerical Feature Distribution Analysis and Transformation**: By plotting the distributions of numerical features (such as `age`, `balance`, `duration`, `campaign`, `pdays`, `previous`), we observed significant right-skewness in multiple features. This skewness could adversely affect the performance of certain machine learning models, especially those sensitive to feature distributions (e.g., linear models, distance-based models).

  ![image-20250515192401748](.\Typora_image\image-20250515192401748.png)								**Figure ?: Distribution of Some Original Numerical Features**

  To address this issue, for right-skewed variables with all values greater than 0 (such as `duration`, `campaign`, `previous`), we considered and attempted a logarithmic transformation (`np.log1p`) to alleviate the skewness and make the data distribution closer to symmetrical. For right-skewed variables containing negative values or special values (such as -1 in `pdays`, negative numbers in `balance`), we adopted a more detailed approach:

  - **`pdays`**: An indicator variable `pdays_was_contacted` (indicating whether they had been contacted before) was created. After processing the -1 values in the original `pdays` (representing "not contacted"), the remaining valid day counts were transformed using `log1p` to generate `pdays_log1p`.

  - **`balance`**: A sign indicator feature `balance_sign_category` (indicating whether the balance was positive, negative, or zero) was created, and the absolute value of the balance was transformed using `log1p` to generate `balance_log_abs_p1`.

    *![image-20250515192507162](.\Typora_image\image-20250515192507162.png)* **Figure 2: Distribution of Some Numerical Features After Transformation**

  As seen in Figure 2, after transformation, the skewness of distributions like `duration`, `campaign`, `previous`, `pdays_log1p`, and `balance_log_abs_p1` was significantly improved.

- **Class Imbalance Problem**: After analyzing the distribution of the target variable `y` (whether the customer subscribed to a term deposit), we found a significant class imbalance. Specifically, the number of negative samples (did not subscribe, y=0) was much larger than the number of positive samples (subscribed, y=1), approximately 7:1 (*please modify according to your actual ratio*). This imbalance is a key issue to address in subsequent modeling.

  - ![image-20250515192101015](.\Typora_image\image-20250515192101015.png)

      						 **Figure ?: Target Variable Class Distribution**

### 2. Baseline Models and Initial Evaluation

To establish a performance baseline and gain an initial understanding of model performance with only basic data cleaning and without addressing class imbalance, we tried several classic classification models. These models included: Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (Linear Kernel), and Gaussian Naive Bayes.

We used standard evaluation metrics such as Accuracy, Precision, Recall, and F1-score to measure model performance.

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

​				**Table ?: Performance of Baseline Models on Unprocessed Skewed and Imbalanced Data**

From Table 1, it can be seen that without adequately addressing data characteristics (especially class imbalance), the performance of various models in terms of Recall and F1-score was generally low, particularly their ability to identify the minority class (subscribing customers).

### 3. Impact of Data Transformation on Model Performance

Next, based on the handling of numerical feature skewness in Section 1 (mainly logarithmic transformation), we retrained the above baseline models to evaluate the improvement in model performance due to data transformation. We call this stage the "log_baseline model".

![image-20250515193234406](.\Typora_image\image-20250515193234406.png)

​						**Figure ?: Performance of Baseline Models on Log-Transformed Data**

Comparing the results of Table 1 and Table 2, we found that logarithmic transformation brought some improvement to the performance of certain models, but the overall improvement was limited. This led us to focus more on the core issue observed earlier—class imbalance.

### 4. Addressing Class Imbalance: SMOTE Oversampling and Model Performance Improvement

#### 4.1 Oversampling and SMOTE

To mitigate the negative impact of class imbalance on model training, a common strategy at the data level is **Oversampling**. The core idea of oversampling is to increase the number of minority class samples, making the proportion of samples from different classes in the training data more balanced, thereby allowing the model to give sufficient attention to the minority class during training.

There are various oversampling techniques, among which **SMOTE (Synthetic Minority Over-sampling Technique)** is a widely used and effective method. Unlike simply randomly duplicating minority class samples (which can lead to overfitting), SMOTE generates new, synthetic minority class samples in the following way:

1.  For each minority class sample point $x_i$.
2.  Find its $k$ nearest neighbors in the minority class samples.
3.  Randomly select one sample point $x_j$ from these $k$ neighbors.
4.  Randomly select a point on the line segment between $x_i$ and $x_j$ as the new synthetic sample. The new sample point $x_{new}$ can be expressed as: $x_{new} = x_i + \delta \cdot (x_j - x_i)$, where $\delta$ is a random number between 0 and 1.

In this way, SMOTE can create new samples that are similar to existing minority class samples but not identical, which helps to expand the decision region of the minority class and improve the model's learning effect on the minority class, while avoiding the overfitting risk brought by simple duplication to a certain extent. However, it is also necessary to note the potential problems of SMOTE introducing noise or blurring class boundaries, especially when minority class samples are very sparse or highly overlapping with other class samples.

#### 4.2 Model Improvement and Exploration

To address the class imbalance problem, we adopted the SMOTE (Synthetic Minority Over-sampling Technique) oversampling method. SMOTE generates new synthetic samples by interpolating between minority class samples, thereby balancing the class distribution in the training set. We applied SMOTE to the log-transformed training data and then retrained each baseline model, calling this stage the "log_smote_baseline model".

Specifically, for the CatBoost model, considering its inherent ability to handle categorical features well and its capability to automatically adjust class weights to address imbalance by setting the `scale_pos_weight` parameter, we compared its performance using SMOTE oversampled data versus using the original data (with only log transformation and categorical feature encoding, and setting the weight adjustment parameter). Preliminary experiments showed that for CatBoost, utilizing its built-in weight adjustment mechanism on the original (or only log-transformed) data performed better. Therefore, subsequent comparisons for CatBoost will be based on its own balancing mechanism.

For most other models, the results after applying SMOTE are as follows:

![image-20250515193931567](.\Typora_image\image-20250515193931567.png)

​					**Figure ?: Performance of Models on Log-Transformed and SMOTE Oversampled Data**

From Table 3, it can be seen that after applying SMOTE oversampling to the training data, most models showed significant improvements in Recall and F1-score. This fully demonstrates the importance of data balancing for improving the model's ability to identify the minority class (i.e., customers who successfully subscribed).

At this stage, we also exploratively borrowed the idea of ensemble learning and attempted to build a VotingClassifier from the three previously strongest performing models .

```python
estimators = [
    ('CatBoost', catboost_model),
    ('XGBoost', xgboost_model),
    ('PyTorch', pytorch_wrapper)
]

voting_clf = VotingClassifier(estimators=estimators, voting='soft')
```

The results showed that the F1-score of the VotingClassifier improved slightly compared to the single best model, but the Recall decreased slightly. More importantly, we noticed a sharp decrease in the Precision metric at this point. This phenomenon prompted us to further optimize the model's decision boundary in conjunction with actual business requirements.

![image-20250515194143505](.\Typora_image\image-20250515194143505.png)

​					**Figure ?: F1-score of All Models**

![image-20250515194219493](.\Typora_image\image-20250515194219493.png)

​					**Figure ?: Recall of All Models**

![image-20250515194918602](.\Typora_image\image-20250515194918602.png)

​						**Figure ?: Comparison of Model Precision**

### 5. Model Optimization with Business Constraints: Maximizing Recall under Precision Constraints

In actual banking marketing scenarios, blindly pursuing recall (i.e., identifying all potential customers) can lead to excessively high marketing costs (as it involves contacting a large number of non-potential customers). Therefore, businesses usually have a minimum requirement for marketing precision. According to the project settings, we introduced the following business constraints:

- Core Objective: Maximize Recall while ensuring Precision ≥ 50%.
  - **Bottom Line (Precision ≥ 50%)**: This means that among the customers our model predicts as "will subscribe" and whom we market to, at least half are indeed potential subscribers. Below this ratio, the marketing campaign's efficiency is considered too low, wasting resources.
  - **Optimization Goal (Recall Maximization)**: After meeting the above Precision bottom line, we want to identify as many of the truly subscribing customers as possible.

To achieve this goal, we selected several models that performed well in the previous stage (e.g., *Random Forest, LGBM, CatBoost, and the previous VotingClassifier*) and, by adjusting the classification threshold of these models' prediction probabilities, sought the best Recall for each model while satisfying the Precision ≥ 50% condition.

![image-20250515195043598](.\Typora_image\image-20250515195043598.png)

​					**Figure ?: Precision-Recall Performance of Some Models After Threshold Adjustment**

|       Model       | Best Threshold |
| :---------------: | -------------- |
|      XGBoost      | 0.626          |
|     CatBoost      | 0.626          |
|    PyTorch NN     | 0.465          |
| Voting Classifier | 0.545          |

​				**Table ?: Best Threshold and Corresponding Recall for Each Model under Precision ≥ 50% Constraint**



| Model                             | Precision | Recall   | F1-score |
| --------------------------------- | --------- | -------- | -------- |
| CatBoost (precision≥50%)          | 0.503756  | 0.811649 | 0.621669 |
| Voting Classifier (precision≥50%) | 0.500234  | 0.809380 | 0.618318 |
| XGBoost (precision≥50%)           | 0.501415  | 0.804085 | 0.617664 |
| PyTorch NN (precision≥50%)        | 0.500995  | 0.761725 | 0.604442 |

​				**Table ?: Optimal Model Metrics After Constraints**

### 6. Future Work and Improvements

Although this project has achieved certain results in predicting bank marketing responses, especially in exploring data skewness, class imbalance, and optimizing models with business constraints, there are still several directions worthy of further in-depth research and improvement to achieve better model performance and stronger practical application value.

1.  **Granular Hyperparameter Tuning**:
    -   In the current work, hyperparameter tuning for each model was mainly based on experience and preliminary experiments. In the future, more systematic and detailed hyperparameter searches can be conducted for well-performing models like LGBM, CatBoost, and potentially promising traditional models (e.g., Random Forest). We plan to use Bayesian optimization tools like Optuna, combined with rigorous cross-validation mechanisms, to maximize target metrics (such as Recall or AUC-PR) while satisfying business constraints (e.g., Precision ≥ 50%), thereby unlocking the full potential of the models.
2.  **Further Exploration of Feature Engineering**:
    -   **Mining Interaction Features**: Although some basic feature transformations have been performed, systematically creating and selecting meaningful interaction features is key to improving model performance. For example, combinations like "age group" with "job category" or "education level" with "has housing loan" can be explored. In addition to business-driven hypotheses, feature importance output from tree-based models (like Random Forest, LGBM) or techniques like polynomial feature generation (`PolynomialFeatures`) can assist in identifying and constructing high-value interaction terms.
    -   **Domain Knowledge-Driven Feature Construction**: By further integrating the characteristics of the banking business, consider whether other customer behavior data in the bank (such as transaction frequency, types of products held, channel preferences, if available) can be transformed into effective predictive features.
3.  **Advanced Strategies for Imbalanced Data**:
    -   **Trying Diversified Sampling Techniques**: This project mainly used SMOTE oversampling. In the future, other advanced sampling algorithms can be further tested and compared. For example, ADASYN (which focuses more on hard-to-learn minority samples), Borderline-SMOTE (which focuses on generating samples near class boundaries) for oversampling; various undersampling methods (like NearMiss, EditedNearestNeighbours, but be mindful of information loss risk); or more complex hybrid sampling methods like SMOTEENN (SMOTE followed by ENN to clean noise) and SMOTETomek (SMOTE followed by Tomek Links to remove sample pairs blurring boundaries). The goal is to find the imbalanced data handling solution best suited to this dataset's characteristics, always evaluating effectiveness under specific business metrics.
    -   **Deepening Cost-Sensitive Learning**: In addition to adjusting class weights, a more detailed cost-sensitive learning framework can be introduced, assigning different costs—more aligned with actual business impact—to different types of misclassifications (e.g., the opportunity loss from misclassifying a "will subscribe" customer as "will not subscribe" versus the marketing cost wasted from misclassifying a "will not subscribe" customer as "will subscribe").
4.  **Optimizing and Exploring Ensemble Strategies**:
    -   **Application of Advanced Ensemble Methods**: Building on the initial attempt with VotingClassifier, more complex ensemble strategies, particularly Stacking, can be explored in the future. By using the prediction results of multiple well-performing and somewhat diverse base models (e.g., LGBM, CatBoost, and possibly a tuned deep learning model or a traditional model) as input features for a meta-learner (e.g., logistic regression or a simple neural network), it is hoped to achieve better ensemble performance than simple voting or averaging. The key lies in carefully selecting base models and tuning the meta-learner.
    -   **Ensuring Diversity of Base Models**: When constructing ensemble models, attention should be paid to the diversity among base models (e.g., they are based on different algorithmic principles or use different feature subsets), which is a prerequisite for the ensemble to surpass the single best model.
5.  **Model Interpretability and Business Insights**:
    -   Although the focus of this project is on predictive performance, enhancing model interpretability is equally important for business application and trust-building. In the future, tools like SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) can be used to deeply analyze the prediction logic of the finally selected model, understand which features are driving predictions, and how they affect individual customer predictions. This can not only provide valuable business insights (e.g., which customer segments are most likely to respond to marketing) but also help discover potential data issues or model biases.

Through further exploration and improvement in the above directions, we expect to build customer response models with better predictive performance, stronger robustness, and that can better serve actual banking business needs.