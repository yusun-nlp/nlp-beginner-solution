## Task4

本次项目主要完成命名实体识别（NER）。**NER系统就是从非结构化的输入文本中抽取出具有特定意义或者指代性强的实体，并且可以按照业务需求识别出更多类别的实体**。

在基于机器学习的方法中，NER被当作**序列标注**问题。利用大规模语料来学习出标注模型，从而对句子的各个位置进行标注。

数据集是CONLL 2003数据集，CoNLL2003中， 实体被标注为四种类型：

- LOC (location, 地名)
- ORG (organisation， 组织机构名)
- PER （person， 人名）
- MISC (miscellaneous， 其他)

用到了ESIM模型

<img src="https://img-blog.csdn.net/20170822121044111?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTEFXXzEzMDYyNQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" alt="这里写图片描述" style="zoom:60%;" />

##### 参考过的网站：

- 深度学习引用于NER：https://www.jiqizhixin.com/articles/2018-08-31-2

##### 相关论文：

- [Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360)

##### 运行方法：

```c++
python train.py
```

##### 文件说明：

- data_helper.py：数据预处理
- model.py：模型定义
- train.py：训练模型



>  代码参考：https://github.com/ZhixiuYe/NER-pytorch

