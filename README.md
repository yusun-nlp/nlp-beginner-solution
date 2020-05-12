# nlp-beginner-solution
:bulb: The solution to nlp-beginner project in FudanNLP

本项目记录FudanNLP的[nlp-beginner](https://github.com/FudanNLP/nlp-beginner)项目的解决代码，代码中进行了较为详细的注释。Readme中列出了学习过程中查阅过的网站，方便复查复看。

## Task1

该任务是基于情感的文本分类，基于烂番茄上的电影评价数据，以五个值的等级来标记短语：否定，有些否定，中性，有些肯定，肯定。是较为典型的文本分类问题。

##### 参考过的网站：

- n-gram模型：https://zhuanlan.zhihu.com/p/32829048
- 梯度下降方法：https://zhuanlan.zhihu.com/p/25765735
- 机器学习调参：https://blog.csdn.net/pipisorry/article/details/52902797

##### 运行方法：

```
python main.py
```

##### 文件说明：

- main.py：主程序
- cache.txt：运行时控制台的输出
- result.csv：选取的5000个测试用例的预测结果



>  代码参考：[https://github.com/KuNyaa/nlp-beginner-implementations/tree/master/%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB](https://github.com/KuNyaa/nlp-beginner-implementations/tree/master/文本分类)



## Task2

### TextCNN的tensorflow实现

TextCNN的结构：输入数据首先通过一个embedding layer，得到输入语句的embedding表示，然后通过一个convolution layer，提取语句的特征，最后通过一个fully connected layer得到最终的输出，整个模型的结构如下图：

<img src="https://img-blog.csdn.net/20180319223936424?watermark/2/text/Ly9ibG9nLmNzZG4ubmV0L3UwMTI3NjI0MTk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="这里写图片描述" style="zoom:50%;" />

##### 参考过的网站

- **TextCNN**：https://blog.csdn.net/chuchus/article/details/77847476
- CNN的参数：[https://flat2010.github.io/2018/06/15/%E6%89%8B%E7%AE%97CNN%E4%B8%AD%E7%9A%84%E5%8F%82%E6%95%B0/](https://flat2010.github.io/2018/06/15/手算CNN中的参数/)
- l1和l2正则化：https://blog.csdn.net/jinping_shi/article/details/52433975?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-7&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-7
- tf.nn.embedding_lookup：https://www.jianshu.com/p/6e61528acad9
- TextCNN模型代码解析：https://blog.csdn.net/u013818406/article/details/69530762
- 训练模型解析：http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

##### 相关论文：

- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

##### 运行方法：

```c
// python data_helper.py
python main.py
```

##### 文件说明

- data_helper.py：数据预处理程序
- main.py：主程序
- cache.txt：运行main.py时控制台的输出
- result.csv：选取的5000个测试用例的预测结果



> 代码参考：https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py