import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow.python import keras
from keras import layers
from keras import losses

# 已经有 test train
# 缺少验证集 val 划分验证集
batch_size = 32
seed = 42
# 经过这一步之后 训练集中有 80% 的数据，验证集中有 20% 的数据
# text_dataset_from_directory 会根据文件夹路径自动加载数据集 标记Label
raw_train_dataset = tf.keras.utils.text_dataset_from_directory(
    'stack_overflow_16k/train',
    batch_size=batch_size,
    validation_split=0.2,  # 以20%的数据作为验证集
    subset='training',  # 从训练集中取
    seed=seed)

# 打印测试 .take(1) 取一个batch
# for text_batch, label_batch in raw_train_dataset.take(1):
#   for i in range(3):
#     print("Article", text_batch.numpy()[i])
#     print("Label", label_batch.numpy()[i])
# 对应标签名称
# for i in raw_train_dataset.class_names:
#     print(i)

# 创建验证数据集和测试数据集
raw_val_dataset = tf.keras.utils.text_dataset_from_directory(
    'stack_overflow_16k/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',  # 从验证集中取
    seed=seed)

raw_test_dataset = tf.keras.utils.text_dataset_from_directory(
    'stack_overflow_16k/test',
    batch_size=batch_size)

# 准备用于训练的数据集
# 创建一个 TextVectorization 层 用于标准化、词例化和向量化
max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    # standardize=custom_standardization, 这一步在原来的数据集中需要，我们这个都是问题的txt
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# lambda x, y: x 是一个匿名函数，它接受两个参数 x 和 y，并只返回 x。
# 在 raw_train_dataset 数据集中，每个元素都是一个 (text, label) 对，其中 text 是文本数据，label 是相应的标签。
# 这个 map 调用实际上是在提取每个 (text, label) 对中的文本部分 text，而忽略了标签 label。
train_text = raw_train_dataset.map(lambda x, y: x)

# train_text是一个tf.data.Dataset对象，其中每个元素都是一个字符串列表，对应于原始数据集中的一个文件。
# <_TakeDataset element_spec=TensorSpec(shape=(None,), dtype=tf.string, name=None)>
# 打印内容
# for text_batch in train_text.take(1):
#     for text in text_batch.numpy():
#         print(text)

# vectorize_layer是我们自定义的 TextVectorization 层，我们可以调用它来向量化我们的数据。
# 当前我们的数据集中的每个元素都是一个字符串列表，因此我们首先调用 adapt 方法来适应数据。
# 通过我们现在的train_text构建词汇表
vectorize_layer.adapt(train_text)


# vectorize_text 函数接受一个字符串列表，返回一个整数张量。
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


"""
迭代器 (iter)：

iter() 函数用于获取一个可迭代对象（如列表、元组、集合、字典等）的迭代器。
在这个例子中，raw_train_dataset 是一个 TensorFlow 数据集对象，它本身是可迭代的。当您调用 iter(raw_train_dataset) 时，您实际上是获取了 raw_train_dataset 的迭代器。
获取下一个元素 (next)：

next() 函数用于访问迭代器的下一个元素。
当调用 next(iter(raw_train_dataset)) 时，您实际上是从 raw_train_dataset 的迭代器中获取了第一个元素（或下一个元素，取决于迭代器的当前状态）。
"""
# text_batch, label_batch = next(iter(raw_train_dataset))
# first_review, first_label = text_batch[0], label_batch[0]
# print("Review", first_review)
# print("Label", raw_train_dataset.class_names[first_label])
# print("Vectorized review", vectorize_text(first_review, first_label))

# 现在我们已经准备好训练数据了。调用 prefetch 方法可以让我们在训练模型的同时在后台获取数据集的批次。
# print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
# print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
# print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

# 准备训练模型
# 最终的数据集都是 (text, label) 对的形式，其中 text 是一个整数张量，label 是一个整数标签。
train_ds = raw_train_dataset.map(vectorize_text)
val_ds = raw_val_dataset.map(vectorize_text)
test_ds = raw_test_dataset.map(vectorize_text)

# text_batch, label_batch = next(iter(train_ds))
# print("text_batch", text_batch[0])
# print("label_batch", label_batch[0])
"""
text_batch: 这是经过 vectorize_text 函数处理的文本数据。在这个批次中，每个文本样本已被转换成一个整数序列，其中每个整数代表一个单词的索引。shape=(250,) 表明每个文本样本被处理成长度为 250 的序列。如果原文本较长，它会被截断；如果较短，会被填充。

label_batch: 这是对应的标签数据。在您的输出中，tf.Tensor(1, shape=(), dtype=int32) 表明当前样本的标签是 1。根据您的数据集，这个标签可能代表某个特定的类别（例如，在情感分析中，0 可能代表负面情绪，1 可能代表正面情绪）。
"""
# text_batch tf.Tensor(
# [1011  773    9 2456    8 1863 2362  690 1267    4   40    5    1 1011
#   196   12   74   13   72   33    2   98  105   14    3   70 9611    3
#    34  888  202  773  107    8   41  242   40   58  291   90    3  196
#   191   10    2  182    6  668    6   13   30 1187   12  773   22   42
#     1   28    5  140   29 5213   15   29    1   28   51    1    1    1
#     7   23   30    3  291   10   67    6   32   65  185  166  102   14
#     2   65    6    1  193    9 2784   45 2410    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0], shape=(250,), dtype=int64)
# label_batch tf.Tensor(1, shape=(), dtype=int32)

# 配置数据集以提高性能
# AUTOTUNE 机器学习中的一种技术，它会在训练时自动调整数据集的配置，以提高性能。
# .cache() 保持数据在内存中，加快读取速度。
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 创建模型
# 使用 Keras Sequential API 构建模型。该模型是层的线性堆叠。
embedding_dim = 16  # 词向量维度
model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),  # +1 是为了防止有UNK
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),  # GlobalAveragePooling1D 将通过对序列维度求平均值来为每个样本返回一个定长输出向量。这允许模型以尽可能最简单的方式处理变长输入
    layers.Dropout(0.2),
    layers.Dense(4)])  # 注意这边是四个分类

model.summary()

# binary_crossentropy 二分类交叉熵 但是我们是多分类 所以使用 SparseCategoricalCrossentropy
model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=[tf.metrics.SparseCategoricalAccuracy()])

# 训练模型
epochs = 10
# history 是一个包含训练过程中损失和指标值的字典
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)


loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

history_dict = history.history
print(history_dict.keys())

# 绘制训练损失和验证损失
acc = history_dict['sparse_categorical_accuracy']
val_acc = history_dict['val_sparse_categorical_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

# 保存模型
# model.save(r'./model/stack_overflow_classification')
# 在模型中包含 TextVectorization 层
