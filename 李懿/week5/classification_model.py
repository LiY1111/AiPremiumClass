import pandas as pd
import fasttext
from sklearn.model_selection import train_test_split

#3. 使用课堂示例cooking.stackexchange.txt，使用fasttext训练文本分类模型。

#读取数据
data = pd.read_csv("cooking.stackexchange.txt", sep='\t', header=None, name=["text"])

#划分数据集
train, test = train_test_split(data, test_size=0.2, random_state=42)

#保存FastText格式
train.to_csv("cooking_train.txt", index=False, header=False)
test.to_csv("cooking_test.txt", index=False, header=False)

model = fasttext.train_supervised(
    input="cooking_train.txt",
    lr=0.1
    epoch=50,
    wordNgrams=2,
    loss='ova'
    verbose=2
)

model.save_model("cooking_model.bin")

result = model.test("cooking_test.txt")
print(f"准确率：{result[1]*100:.2f}%")

text = "How to bake bread with whole wheat flour?"
labels, probs = model.predict(text, k=2)
print(f"预测标签：{labels}, 概率: {probs}")
