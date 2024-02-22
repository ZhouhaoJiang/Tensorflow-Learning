import tensorflow as tf

print(tf.__version__)
# exit()

# tf.keras.datasets 是 TensorFlow 2.0 版本新增的模块，专门用于下载数据集
# mnist 是手写数字数据集，包含 60000 张 28x28 的训练样本和 10000 张测试样本
mnist = tf.keras.datasets.mnist

# x_train 和 x_test 代表样本，y_train 和 y_test 代表标签
# 除以 255 是为了做归一化，将像素值缩放到 0 到 1 之间
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# tf.keras是TensorFlow 2.0版本新增的高阶API，用于快速构建和训练深度学习模型
# tf.keras.models.Sequential 是一系列网络层按顺序构成的栈
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # 将 28x28 的像素值矩阵展平成 784 的向量 Flatten 层用于展平张量
    tf.keras.layers.Dense(128, activation='relu'),  # 全连接层，128 个节点，激活函数为 relu，relu是一个线性整流函数
    tf.keras.layers.Dropout(0.2),  # Dropout 层，防止过拟合 Dropout是一种正则化方法，通过在训练过程中随机丢弃神经元，来减少神经元之间的联合适应性
    tf.keras.layers.Dense(10, activation='softmax')  # 全连接层，10 个节点，激活函数为 softmax，softmax 是一个归一化指数函数
])

predictions = model(x_train[:1]).numpy()  # 预测x_train中第一个样本的输出 转换为numpy数组
print(predictions)
# [[0.05086253 0.08202326 0.07281421 0.07065959 0.12530726 0.10585878
#   0.1041206  0.12408201 0.1699522  0.09431953]]

# 优化器（Optimizer）：决定如何基于损失函数对网络进行更新
model.compile(optimizer='adam',  # Adam 是一种自适应学习率的优化算法，它能够根据训练过程动态调整学习率
              loss='sparse_categorical_crossentropy',  # 损失函数（Loss Function）：衡量网络输出和标签之间的距离
              metrics=['accuracy'])  # 指标（Metrics）：监控训练和测试步骤

# 训练模型 model.fit() 方法会返回一个 History 对象，它包含了训练过程中的 loss 和 metrics
model.fit(x_train, y_train, epochs=5)
# 评估模型
model.evaluate(x_test, y_test)

# 模型推理
# 使用 tf.keras.Sequential加载了model，这个model包含了训练好的网络结构和参数
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()  # Softmax层将模型的原始输出转换为概率
])

print(probability_model(x_test[:5])) # 预测x_test中前5个样本的输出
# tf.Tensor(
# [[0.0853401  0.08534008 0.0853413  0.08536153 0.08534008 0.08534008
#   0.08534008 0.2319165  0.0853401  0.08534017]
#  [0.08534549 0.0853456  0.23183079 0.08540485 0.08534549 0.08534551
#   0.08534549 0.08534549 0.08534575 0.08534549]
#  [0.08534543 0.23183197 0.08536094 0.08534788 0.08534665 0.08534566
#   0.08534583 0.08537371 0.08535634 0.08534553]
#  [0.23194915 0.08533802 0.08534058 0.08533803 0.08533802 0.0853394
#   0.08534138 0.08533883 0.08533801 0.08533853]
#  [0.0855104  0.08550816 0.08551189 0.0855083  0.22923432 0.08550839
#   0.08551147 0.0856301  0.08550841 0.08656855]], shape=(5, 10), dtype=float32)
