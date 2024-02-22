# pip install tensorflow-hub
# pip install tensorflow-datasets
# 用 tf.keras（一个在 TensorFlow 中用于构建和训练模型的高级 API）和 tensorflow_hub（一个用于在单行代码中从 TFHub 加载训练模型的库

import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# 打印版本信息
# print("Version: ", tf.__version__)
# print("Eager mode: ", tf.executing_eagerly())
# print("Hub version: ", hub.__version__)
# print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

# 下载数据集
# Split the training set into 60% and 40% to end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'), # 60% 40% 0%
    as_supervised=True,
)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
# print(train_examples_batch)
# print(train_labels_batch)

# 使用文本嵌入向量模型
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
# print(hub_layer(train_examples_batch[:3]))

# 构建模型
model = tf.keras.Sequential() # Sequential模型是多个网络层的线性堆叠
model.add(hub_layer) # 添加嵌入层 Embedding layer
model.add(tf.keras.layers.Dense(16, activation='relu')) # 添加全连接层 Dense layer
model.add(tf.keras.layers.Dense(1)) # 添加全连接层 Dense layer Dense(1)表示输出层只有一个节点
print(model.summary()) # 打印模型概述
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  keras_layer (KerasLayer)    (None, 50)                48190600
#
#  dense (Dense)               (None, 16)                816
#
#  dense_1 (Dense)             (None, 1)                 17
#
# =================================================================
# Total params: 48191433 (183.84 MB)
# Trainable params: 48191433 (183.84 MB)
# Non-trainable params: 0 (0.00 Byte)
# _________________________________________________________________

# 损失函数与优化器
model.compile(optimizer='adam', # 优化器 adam用于自适应学习率的优化算法 它能够根据训练过程动态调整学习率 此外还有SGD RMSprop等
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # 二分类交叉熵 多分类用 CategoricalCrossentropy
              metrics=['accuracy']) # 指标（Metrics）：监控训练和测试步骤

# 训练模型
# 使用包含 512 个样本的 mini-batch 对模型进行 10 个周期的训练，
# 也就是在 x_train 和 y_train 张量中对所有样本进行 10 次迭代。在训练时，监测模型在验证集的 10,000 个样本上的损失和准确率：
epochs = 10
history = model.fit(train_data.shuffle(10000).batch(512), # .shuffle(10000) 将数据集打乱，每次训练时都会打乱数据集
                    epochs=epochs,
                    validation_data=validation_data.batch(512), # .batch(512) 将数据集划分为大小为 512 的 batch
                    verbose=1) # verbose=1 显示进度条 verbose=0 不显示进度条

# 评估模型
results = model.evaluate(test_data.batch(512), verbose=2) # verbose=2 显示进度条 verbose=0 不显示进度条

# zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
# model.metrics_names 为 ['loss', 'accuracy']
# results 为 [0.335, 0.855]
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))




