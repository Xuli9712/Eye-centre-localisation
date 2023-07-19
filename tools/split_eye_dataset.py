import json
from sklearn.model_selection import train_test_split

# 读取数据
with open(r'F:\gi4e_database\blend_eye_pt.json', 'r') as f:
    data = json.load(f)

# 划分数据集
train, test = train_test_split(data, test_size=0.2, random_state=42)  # 30% 的数据作为测试集 + 验证集
test, val = train_test_split(test, test_size=0.5, random_state=42)  # 剩下的 30% 数据中的 50% 作为验证集和测试集，分别占15%

# 写入新的json文件
with open('data_train.json', 'w') as f:
    json.dump(train, f)

with open('data_val.json', 'w') as f:
    json.dump(val, f)

with open('data_test.json', 'w') as f:
    json.dump(test, f)
