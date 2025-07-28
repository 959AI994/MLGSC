# Multi-level Graph Subspace Contrastive Learning for Hyperspectral Image Clustering

**Authors**: Jingxin Wang, Renxiang Guan, Kainan Gao, Zihao Li, Hao Li, Xianju Li, Chang Tang

## Abstract
Hyperspectral image (HSI) clustering is a challenging task due to its high complexity. Despite subspace clustering showing impressive performance for HSI, traditional methods tend to ignore the global-local interaction in HSI data. In this study, we proposed a multi-level graph subspace contrastive learning (MLGSC) for HSI clustering. The model is divided into the following main parts:

1. **Graph convolution subspace construction**: Utilizing spectral and texture features to construct two graph convolution views.
2. **Local-global graph representation**: Local graph representations were obtained by step-by-step convolutions, and a more representative global graph representation was obtained using an attention-based pooling strategy.
3. **Multi-level graph subspace contrastive learning**: Multi-level contrastive learning was conducted to obtain local-global joint graph representations, to improve the consistency of the positive samples between views, and to obtain more robust graph embeddings.

Specifically, graph-level contrastive learning is used to better learn global representations of HSI data. Node-level intra-view and inter-view contrastive learning is designed to learn joint representations of local regions of HSI.

The proposed model is evaluated on four popular HSI datasets: Indian Pines, Pavia University, Houston, and Xu Zhou. The overall accuracies are 97.75%, 99.96%, 92.28%, and 95.73%, which significantly outperforms the current state-of-the-art clustering methods.

## Paper Link
You can access the full paper on [arXiv: Multi-level Graph Subspace Contrastive Learning for Hyperspectral Image Clustering](https://arxiv.org/abs/2404.05211)

## Results
- **Indian Pines**: 97.75% accuracy
- **Pavia University**: 99.96% accuracy
- **Houston**: 92.28% accuracy
- **Xu Zhou**: 95.73% accuracy

These results significantly outperform current state-of-the-art clustering methods.

## Keywords
- Hyperspectral Image Clustering
- Graph Convolution
- Contrastive Learning
- Multi-level Learning
  
## 注意事项
### 用法
`./old/main.py`
> 废案，当作分类任务的代码

`main.py`
> 废案，使用了过于复杂的对比学习方法，效果也不佳

`main2.py,main5,main6`
> 少量数据点用这个，会对对所有点构建大图，一般比用了minibatch的精度高。

`main3.py`
> 大量数据点用这个，会用到`batch_size`参数，降低精度，训练速度快。而且对有大量数据点的数据集，还可能可能得设置`eval_cluster`参数为`kmeans`

训练命令如下，main2的位置可以替换为main3，训练集名称分别为`['Indian_pines', 'paviaU', 'Houston', 'xuzhou']`，区分大小写。

``` bash
python main4.py                         # 直接训练，默认训练集为Indian_pines
python main4.py Indian_pines            # 在Indian_pines数据集上<训练>并<测试>，训练过程和测试结果保存到./result/，模型保存到./save/
python main4.py Indian_pines model.pkl  # 在Indian_pines数据集上加载对应数据集已保存的模型进行<测试>，测试结果保存到./result/
```

### config部分参数的解释，部分数据集测试时所用的值已标出
`train_cluster`
> 训练时用的聚类方法，可选 `kmeans`、`gcsc`、`gcsck`，对模型没有影响，只是用来看的，方便**手动**调参。基本上kmeans就ok了，用gcsc看的更清楚，但跑的慢。

`eval_cluster`
> 测试时用的聚类方法，可选 `kmeans`、`gcsc`、`gcsck`，对模型没有影响。测试时用gcsc聚类效果一般会比kmeans好（如果测试时用gcsck可不一定）

`batch_size`
> 有的数据集点太多，必须用minibatch，就会用到这个参数

`num_pc`
> PCA提取维度，取4已经包含96%以上的数据了。

`size`
> Spectral feature的邻域大小，**奇数**。

`num_openings_closings`
> emp特征提取的参数，调节算法的开闭运算，可以不用动。

`n_neighbors`
> KNN构建邻接矩阵的K。邻接矩阵构建方法同图子空间聚类，看代码是根据空间距离构建的，非余弦相似度。
+ indian: 30
+ paviaU: 20

`drop1, drop2`
> 对比学习参数，本来应该是4个，我给合并成两个了。对比学习时分别对原图和对应邻接矩阵按两种不同drop进行增强。
+ indian: [0.3, 0.5]
+ paviaU: [0.25, 0.4]

### 其他
为了测试较快，和图子空间聚类一样直接取了一小块来测试。如果取全图测试会出现点过多建图过慢、**聚类时间过长**和**耗费内存过多**的问题。
+ indian: 10000+个点，这个可以直接建图。
+ paviaU: 40000+个点，这个不可能直接建图。
+ Houston: 15000+个点，这个勉强可以。
+ xuzhou: 60000+个点，这个不可能直接建图。

---
### 一些简单测试结果
其中，emp特征和现模型用的是GCSC来聚类的，GCSC和GCSCK是图子空间聚类的两种聚类方法。现模型目前仅针对GCSC进行了简单实验，能够比GCSC更好。（GCSCK算的过慢，调起来用时太多）

为啥只测了截取的？所有点全输入的情况下图子空间聚类的两种聚类方法全都太慢了，没办法测代码。

model名后面标`m`意为用了minibatch，标`g`意为用了GCSC来聚类

#### indian截取

| model | OA | Kappa | NMI |
| ------ | ------ | ------ | ------ |
| emp(g) | 0.61626 | 0.45588 | 0.47034 |
| GCSC | 0.88271 | 0.69757 | 0.83081 |
| GCSCK | **0.97313** | **0.91281** | **0.96148** |
| 现模型(g) | 0.91232 | 0.76364 | 0.87428 |
| 现模型(mg) | 0.69460 | 0.50858 | 0.52537 |

#### paviaU截取

| model | OA | Kappa | NMI |
| ------ | ------ | ------ | ------ |
| emp(g) | 0.73887 | 0.81606 | 0.68196 |
| GCSC | 0.70194 | 0.83818 | 0.64112 |
| GCSCK | 0.95733 | 0.95915 | 0.94384 |
| 现模型(g) | **0.95779** | **0.96044** | **0.94458** |
| 现模型(mg) | 0.89681 | 0.91983 | 0.86308 |
