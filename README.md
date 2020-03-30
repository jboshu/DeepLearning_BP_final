# DeepLearning_BP_final
bp算法 服装识别问题

解决方案设计
利用BP全连接网络，将服装数据fashion-MNIST中60000个数据数字化
通过全连接网络进行训练，最后用训练出的网络对剩下10000个数据测试，得到准确率。

数据准备
Fashion-MNIST[3]是Zalando文章图像的数据集-包含60,000个示例的训练集和10,000个示例的测试集。
每个示例都是一个28x28灰度图像，与来自10个类别的标签相关联。
我们打算Fashion-MNIST直接替代原始MNIST数据集，以对机器学习算法进行基准测试。它具有相同的图像大小以及训练和测试分割的结构。
训练数据集为一个28*28的矩阵，训练标签为一个1*10的矩阵，数据量为60000个。
测试数据集为一个28*28的矩阵，训练标签为一个1*10的矩阵，数据量为10000个。
a.	先从数据集中读入数据分别到x_train，x_test，将28*28的矩阵变为784*1的矩阵；
b.	将uint8类型转化为float32类型；
c.	将矩阵中数值转到到[0,1]区间；
d.	将Label中数值转化到[10,1]的矩阵中。

网络总体准确率能够达到基于CNN分类识别准确率[4][5]，但测试时会产生较大波动，经过分析猜测，应该是缺少Dropout来防止过拟合现象
于是加入Dropout后再进行了训练。
经过20次迭代得到最终测试的loss为0.39，acc为0.8831。
