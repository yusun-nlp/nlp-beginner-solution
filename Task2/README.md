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

- [Convolutional Neural Networks for Sentence Classification (Kim Y.)](https://arxiv.org/abs/1408.5882)

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



### TextRNN的Pytorch实现

**流程**：embedding--->Bi-LSTM--->concat final output/average all output----->softmax layer

<img src="https://pic1.zhimg.com/80/v2-7242182ad098259fbbfd1573447cd4d0_1440w.jpg" alt="img" style="zoom:50%;" />

##### 参考网站：

- pytorch的损失函数：https://zhuanlan.zhihu.com/p/61379965
- pytorch实现LSTM：https://www.pytorchtutorial.com/pytorch-sequence-model-and-lstm-networks/
- Pytorch文档：https://pytorch-cn.readthedocs.io/zh/latest/

##### 数据

数据集采用gaussic的数据集，https://github.com/gaussic/text-classification-cnn-rnn
链接: https://pan.baidu.com/s/1hugrfRu 密码: qfud

##### 运行方法：

```c
python train.py  //train model
python test.py	 //make prediction
```

##### 相关论文：

- [Recurrent Neural Network for Text Classification with Multi-Task Learning (Liu P, Qiu X, Huang X, et al.)](https://arxiv.org/abs/1605.05101)

##### 文件说明

- data_helper.py：数据相关的函数
- model.py：TextRNN模型
- train.py：训练程序
- test.py：测试程序

> 代码参考：https://github.com/Alic-yuan/nlp-beginner-finish