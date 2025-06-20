Here’s the refined academic table with proper LaTeX-style mathematical notation ($x$ for variables/equations):

---

#### **Table 3.1: Theoretical Comparison of Selected Models**  

| **Model**               | **Key Formulation**                                                                 | **Advantages**                          | **Limitations**                          | **Selection Rationale**                     |
|-------------------------|------------------------------------------------------------------------------------|-----------------------------------------|------------------------------------------|---------------------------------------------|
| **Logistic Regression** | $P(y=1 \vert \mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$       | Interpretable, efficient computation   | Linear decision boundary only           | Baseline for linear separability analysis   |
| **Decision Tree**       | Split via IG/Gini: $\text{Gain}(D,a) = H(D) - \sum_v \frac{|D^v|}{|D|} H(D^v)$    | Handles nonlinearity, visualizable     | Prone to overfitting                    | Exploratory analysis of feature hierarchies |
| **Random Forest**       | Ensemble of $M$ trees: $\hat{f}(\mathbf{x}) = \text{majority vote}(f_i(\mathbf{x}))$ | Robust to overfitting, parallelizable  | Computationally expensive               | Improves generalizability over single trees |
| **SVM (Linear Kernel)** | $\min_{\mathbf{w},b} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_i \xi_i$ s.t. $y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i$ | Maximizes margin, works in high-dim    | Sensitive to class imbalance            | Validates linear separability hypothesis    |
| **Gaussian Naive Bayes** | $P(y=k \vert \mathbf{x}) \propto P(y=k) \prod_i P(x_i \vert y=k)$                | Fast training, low variance             | Strong independence assumptions         | Benchmark for generative vs. discriminative |
| **XGBoost**            | $\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)$, $\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \|\mathbf{w}\|^2$ | High accuracy, handles missing data    | Hyperparameter-sensitive               | State-of-the-art for imbalanced classification |
| **CatBoost**           | Ordered boosting + categorical feature encoding (e.g., Target Encoding)            | Automatic categorical handling, robust  | Slower than LightGBM                    | Optimized for categorical features (e.g., `job`, `education`) |
| **MLP (PyTorch)**      | $\mathbf{h} = \sigma(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1)$, $\hat{y} = \text{Sigmoid}(\mathbf{W}_2\mathbf{h} + \mathbf{b}_2)$ | Captures complex nonlinearities        | Data-hungry, prone to overfitting       | Tests deep learning performance ceiling    |
| **LightGBM**           | Leaf-wise growth + histogram-based splitting                                       | Extremely fast, memory-efficient       | May overfit without depth constraints   | Scalability for large dataset (>40K samples) |

**Abbreviations**:  
- **IG**: Information Gain, **SVM**: Support Vector Machine, **MLP**: Multilayer Perceptron.  
- **Regularization terms**: $C$=SVM penalty, $\gamma$=XGBoost leaf penalty, $\lambda$=L2 weight decay.  

---

### **Key Adjustments**:  
1. **Consistent LaTeX Notation**: All variables/equations wrapped in `$...$` (e.g., $\mathbf{w}^T\mathbf{x}$).  
2. **Fixed Formatting**: Corrected misplaced braces (e.g., $\|\mathbf{w}\|^2$ instead of `\|\mathbf{w}\|^2`).  
3. **Clarity**: Maintained alignment between mathematical rigor and tabular brevity.  

Let me know if any further refinements are needed!