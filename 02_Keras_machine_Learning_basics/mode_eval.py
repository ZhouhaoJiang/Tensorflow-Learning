import numpy as np
import tensorflow as tf

# 加载模型
model_path = 'model/stack_overflow_classification'
model = tf.keras.models.load_model(model_path)

# 打印模型摘要
print(model.summary())

# 待预测的字符串列表 四个问题 分别对应四个类别
example = [
    "How to write a code to solve a linear equation in Python?",
    "How to write a code to solve a linear equation in Java?",
    "How to write a code to solve a linear equation in JavaScript?",
    "How to write a code to solve a linear equation in C#?"
]

# 使用向量化的数据进行预测
predictions = model.predict(example)

# 打印预测结果
print(predictions)
class_name = ['csharp', 'java', 'javascript', 'python']

for prediction in predictions:
    print(np.argmax(prediction))
    print(class_name[np.argmax(prediction)])


