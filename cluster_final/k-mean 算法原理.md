# k-mean 算法

K-means 算法通过不断交替执行“赋值步骤”（Assignment Step）和“更新步骤”（Update Step），在每次迭代中都会使目标函数——所有样本到所属簇中心的平方距离之和（potential function）——不增。具体来说：

- **赋值步骤**：固定簇中心，将每个点分配到最近的中心，必然不会增大总距离。  
- **更新步骤**：固定分配结果，将每个簇中心更新为其所属点的均值，根据最小二乘原理，该步骤同样不会增大总距离。  

由于 potential function 下界为 0，且每步单调不增，算法最终必收敛于某个局部最优解。

---

## 算法简介

给定数据集 ${x_1, x_2, \dots, x_n}\subset\mathbb{R}^d$和簇数 $k$，K-means 的目标是寻找簇划分 $C = \{C_1,\dots,C_k\}$ 及中心 $\{\mu_1,\dots,\mu_k\}$，以最小化

$$
L(C,\mu) \;=\; \sum_{j=1}^k \sum_{x_i\in C_j} \|x_i - \mu_j\|^2.
$$

**算法流程**（Lloyd 算法）：

1. 初始化 \(k\) 个中心 $\mu_1^{(0)},\dots,\mu_k^{(0)}$。  
2. 对于迭代次数 \(t = 0,1,2,$\dots$\)：  
   - **赋值步骤**（Assignment）：  
     $$
       C_j^{(t+1)} \;=\; \bigl\{\,x_i : j = \arg\min_{\ell}\|x_i - \mu_\ell^{(t)}\|^2 \bigr\}.
     $$
   - **更新步骤**（Update）：  
     $$
       \mu_j^{(t+1)} \;=\; \frac{1}{|C_j^{(t+1)}|}\sum_{x_i\in C_j^{(t+1)}} x_i.
     $$
   - 若簇划分或中心不再改变，则停止。  

---

## potential function 的定义

在文献中，常将  
$$
  \phi(\mu, C) \;=\; \sum_{j=1}^k \sum_{x_i\in C_j} \|x_i - \mu_j\|^2
$$
称为 **potential function** 或 **within-cluster sum of squares (WCSS)**。该函数始终非负，其物理意义是数据点到其对应质心的总平方距离（SSE）。

---

## 为什么 potential function 随迭代单调下降

K-means 的每次迭代包含两个子步骤，每步都不会增加 $\phi$，因此整体单调不增。

### 赋值步骤保证不增

固定中心 $\{\mu_j^{(t)}\}$，对每个点 \($x_i$\) 选择使 $\|x_i - \mu_j^{(t)}\|^2$ 最小的簇：

$$
\phi\bigl(\mu^{(t)},\,C^{(t+1)}\bigr)
= \sum_i \min_j \|x_i - \mu_j^{(t)}\|^2
\;\le\;
\sum_i \|x_i - \mu_{C^{(t)}(i)}^{(t)}\|^2
= \phi\bigl(\mu^{(t)},\,C^{(t)}\bigr).
$$

即，每次重新分配后，不会增大总距离。

### 更新步骤保证不增

固定簇分配 $C^{(t+1)}$，对每个簇求最优中心——所属点的算术均值：

$$
\mu_j^{(t+1)} \;=\; \arg\min_{\mu}\sum_{x_i\in C_j^{(t+1)}}\|x_i - \mu\|^2.
$$

根据最小二乘原理，质心取均值可最小化该簇内的平方误差，因此有

$$
\phi\bigl(\mu^{(t+1)},\,C^{(t+1)}\bigr)
\;\le\;
\phi\bigl(\mu^{(t)},   \,C^{(t+1)}\bigr).
$$

---

## 收敛性结论

- **单调性**：结合上述两步，
  $$
    \phi\bigl(\mu^{(t+1)},C^{(t+1)}\bigr)
    \;\le\;
    \phi\bigl(\mu^{(t)},C^{(t)}\bigr).
  $$
- **有界性**：$\phi \ge 0$
- **状态有限**：每次迭代的 $(C,\mu)$组合有限（中心必须在数据点的凸包内，分配方案有限）。  

因此算法必在有限步内达到不再改变的状态，即收敛到某个局部最优（或鞍点）。





# k-mean算法对初始点选取的敏感性（缺陷）

K-means 算法通过交替执行“最优分配”和“最优更新”两步，确保每次迭代都单调降低目标函数（potential function），并因函数下界和状态有限而必然收敛。尽管收敛,但是收敛到的结果可能只是局部最优。

例子：

![image-20250519165827849](C:\Users\苏政逸\AppData\Roaming\Typora\typora-user-images\image-20250519165827849.png)

![image-20250519165841321](C:\Users\苏政逸\AppData\Roaming\Typora\typora-user-images\image-20250519165841321.png)



# k-mean ++对于初始点选取的改进

K-means++ 是一种改进的初始化方法，通过合理选取初始中心，避免随机初始化带来的聚类效果不稳定问题。其核心思想是：先随机选取一个中心，然后按“距离平方”权重依次选取后续中心，使新中心更可能分布在数据稀疏或远离已有中心的区域，从而加速收敛并提升聚类质量。

---

## 算法目标

给定数据集  
$$
\{x_1, x_2, \dots, x_n\}\subset\mathbb{R}^d
$$
和簇数 \(k\)，K-means++ 的目标是在正式运行 K-means 前，为后续迭代选出一组优质的初始中心  
$$
\{\mu_1, \mu_2, \dots, \mu_k\},
$$
以期最小化最终的总平方误差。

---

## 初始选点规则

1. **选取第一个中心 $\mu_1$**  
   从所有数据点中均匀随机选择一个：  
   $$
   \mu_1 \;\sim\;\text{Uniform}\{x_1,\dots,x_n\}.
   $$

2. **选取第 $j$ ($2 ≤ j ≤ k$) 个中心**  
   
   - 对每个点 \(x_i\)，计算它到已选中心集合 $\{\mu_1,\dots,\mu_{j-1}\}$ 中最近中心的平方距离：  
     $$
     D(x_i) \;=\;\min_{1\le \ell < j}\bigl\|x_i - \mu_\ell\bigr\|^2.
     $$
   - 以概率  
     $$
     P(x_i) \;=\;\frac{D(x_i)}{\sum_{r=1}^n D(x_r)}
     $$
  选出新的中心 $\mu_j$。
   
3. **重复直到选出 \(k\) 个中心** 
   重复第 2 步，直到获得 $\mu_1,\mu_2,\dots,\mu_k $。

---

## 直观与优势

- **降低坏初始化的概率**：相比纯随机初始化，K-means++ 减少了选择相邻簇心和生成空簇的风险；  
- **理论保证**：在期望意义下，该方法可将最终的 SSE（总平方误差）相比随机初始化改善 $O(\log k)$ 倍（参考：Arthur, D. & Vassilvitskii, S. (2007). *k-means++: The Advantages of Careful Seeding*.  ）。

---

## 与标准 K-means 的结合

完成 $k$ 个初始中心选取后，直接按标准 Lloyd 算法执行以下迭代直至收敛：

1. **赋值步骤**：将每个 $x_i$分配到最近的 $\mu_j$；  

2. **更新步骤**：对每个簇 $C_j$计算新中心  
   $$
   \mu_j = \frac{1}{|C_j|}\sum_{x_i\in C_j} x_i.
   $$

合理的初始中心通常能显著减少迭代次数，并提升最终聚类效果。



# k-prototye 对于分类型变量数据的聚类

$K-Prototypes$ 是一种用于混合型数据（包含数值型和分类型变量）的聚类算法，它结合了 $K-Means$（用于数值型）和 $K-Modes$（用于分类型）的思想。核心在于对对象间距离的定义：数值部分使用欧氏距离，分类部分使用简单匹配不相似度，并通过权重系数$\gamma$ 平衡两者影响。

---

## 算法目标

给定包含数值属性和分类属性的数据集  
$$
\{\,x_i = (x_i^{(\text{num})},\,x_i^{(\text{cat})})\in \mathbb{R}^p \times \mathcal{A}_1\times\cdots\times\mathcal{A}_q \mid i=1,\dots,n\},
$$
以及簇数 $k$，K-Prototypes 的目标是寻找簇划分 $\{C_1,\dots,C_k\}$ 及原型$prototype$  
$$
\mu_j = \bigl(\mu_j^{(\text{num})},\,\mu_j^{(\text{cat})}\bigr)
$$
使得混合距离总和最小：

$$
L = \sum_{j=1}^k \sum_{x_i \in C_j} d\bigl(x_i,\;\mu_j\bigr).
$$

---

## 混合型距离度量

对于样本 $x = (x^{(\text{num})},x^{(\text{cat})})$ 与原型 $\mu=(\mu^{(\text{num})},\mu^{(\text{cat})})$，定义距离为

$$
d(x,\mu)
=
\underbrace{\sum_{l=1}^{p} \bigl(x_l^{(\text{num})} - \mu_l^{(\text{num})}\bigr)^2}_{\text{数值部分 (欧氏)}} 
\;+\;
\gamma \;\underbrace{\sum_{m=1}^{q} \delta\bigl(x_m^{(\text{cat})},\,\mu_m^{(\text{cat})}\bigr)}_{\text{分类部分 (不相似度)}},
$$

其中  
- $delta(a,b)=0$ 若 $a=b$，否则 $\delta(a,b)=1$；  
- $\gamma>0$ 为平衡参数，用于调节分类变量与数值变量之间的相对权重。

---

## K-Prototypes 算法流程

1. **初始化**  
   
- 随机或基于抽样选取 $k$个初始原型 $\{\mu_j\}$，每个原型包含一组数值均值和分类众数。  
   
2. **迭代直到收敛**  
   
   - **赋值步骤**： 
     对每个样本 $x_i$，计算其与各原型的混合距离 $d(x_i,\mu_j)$，并分配到距离最小的簇：  
     $$
     C_j = \bigl\{\,x_i : j = \arg\min_{\ell} d(x_i,\mu_\ell)\bigr\}.
     $$
   - **更新步骤**： 
     对每个簇 $C_j$ 分别更新原型：  
     
     - 数值部分：对每个数值变量 $l$ 取簇内样本的算术均值  
       $$
       \mu_{j,l}^{(\text{num})}
       = \frac{1}{|C_j|}\sum_{x_i\in C_j} x_{i,l}^{(\text{num})}.
       $$
     - 分类部分：对每个分类变量 $m$取簇内样本的**众数**  
       $$
       \mu_{j,m}^{(\text{cat})}
       = \arg\max_{a\in \mathcal{A}_m}
      \sum_{x_i\in C_j} \mathbf{1}(x_{i,m}^{(\text{cat})}=a).
       $$
   
3. **收敛判定** 
   当簇分配或原型均不再变化时停止。

---

## 分类变量距离直观

- **简单匹配不相似度**：若样本与原型在某个分类属性上取值相同，则该属性贡献距离 0；否则贡献 1。  
- **权重调节**：参数 $\gamma$控制**每个**分类属性不相似对整体距离的影响，可根据属性取值总数或经验调整，一般取  
  $$
  \gamma \approx \frac{\text{平均数值距离}}{\text{平均分类不相似数}}.
  $$

---

## 收敛性与应用

每次迭代赋值与更新都使目标函数 $L$ 不增，且 $L\ge0$，簇原型和划分状态有限，故算法在有限步内收敛到局部最优。  





