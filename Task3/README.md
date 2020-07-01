## Task3

本次项目主要完成自然语言推断（NLI）。NLI任务主要是关于给定前提premise和假设hypothesis，要求判断p和h的关系。二者的关系有三种：

1. 不相干 neural
2. 蕴含 entailment，即能从p推断出h或两者表达的是一个意思。
3. 冲突 contradiction，即p和h有矛盾

数据集是Stanford Natural Language Inference (SNLI) corpus数据集，用到了ESIM模型

![img](https://pic1.zhimg.com/80/v2-37805edc19fefaa0e1f993ce57c034ec_1440w.jpg)

##### 参考过的网站：

- embedding文件：https://www.kaggle.com/watts2/glove6b50dtxt/data
- ESIM介绍：https://zhuanlan.zhihu.com/p/138158836
- torch.nn相关函数：https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/

##### 相关论文：

- [Enhanced LSTM for Natural Language Inference](https://arxiv.org/abs/1609.06038)

##### 运行方法：

```c++
python data_helper.py  //数据预处理
python train.py
```

##### 文件说明：

- data_helper.py：数据预处理
- layers.py：模型各层的类定义
- model.py：模型定义
- train.py：训练模型
- config.json：参数的存储

由于数据过大，并未上传数据预处理产生的结果和训练结果，可自行运行产生。数据集以zip格式放在相应的文件夹下，解压到当前文件即可使用。



>  代码参考：https://github.com/htfhxx/nlp-beginner_solution/tree/master/Task3

