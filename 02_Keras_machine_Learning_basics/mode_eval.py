# 导入模型
from tensorflow.python import keras
import tensorflow as tf
from keras import layers

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    # standardize=custom_standardization, 这一步在原来的数据集中需要，我们这个都是问题的txt
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)


model = keras.models.load_model(r'./model/stack_overflow_classification')
example = ["how to write multi-threaded code in python?"]
example_vectorized = vectorize_layer(example)
print(model.predict(example))