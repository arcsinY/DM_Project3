import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from catboost import CatBoostRegressor
import math as math

data = pd.read_csv('vgsales.csv')
# 去重
name2id = {}
for i in range(len(data)):
    if data.loc[i, 'Name'] in name2id:
        id = name2id[data.loc[i, 'Name']]
        data.loc[id, 'NA_Sales'] += data.loc[i, 'NA_Sales']
        data.loc[id, 'JP_Sales'] += data.loc[i, 'JP_Sales']
        data.loc[id, 'EU_Sales'] += data.loc[i, 'EU_Sales']
        data.loc[id, 'Other_Sales'] += data.loc[i, 'Other_Sales']
        data.loc[id, 'Global_Sales'] += data.loc[i, 'Global_Sales']
    else:
        name2id[data.loc[i, 'Name']] = i

data.drop_duplicates(subset = ['Name'], keep = 'first', inplace = True)
data.sort_values(by = 'Global_Sales')
data.reset_index(drop=True, inplace = True)

# 缺失值
for i in range(len(data)):
    for j in data.columns:
        if pd.isnull(data.loc[i, j]):
            data.drop(i, inplace = True)
            break
data.reset_index(drop = True, inplace = True)
for i in range(len(data)):
    data.loc[i, 'Rank'] = int(i + 1)
print(data)
# 相关系数
plt.figure(figsize=(11, 9),dpi=100)
sns.heatmap(data = data.drop(['Rank', 'Year'], axis = 1).corr(), annot = True, fmt = ".2f")
plt.show()  

# 数据描述
# 统计标称属性attr各个类别出现频数，返回dict{类别：频数}
def nominal_attr_statistic(data, attr):
    res = {}
    for i in data[attr]:
        if pd.isnull(i):
            continue
        if i in res:
            res[i] += 1
        else:
            res[i] = 1
    return res
# 标称属性柱状图绘制
def nominal_barchart(dict, attr, bound = 0):
    key = []
    value = []
    # 找出满足要求的key和value
    key = []
    value = []
    for k,v in dict.items():
        if v >= bound:
            key.append(k)
            value.append(v)
    key = np.array(key)
    value = np.array(value)
    # 绘制图像
    fig, ax = plt.subplots()
    bar_chart = ax.bar(key, value)
    ax.set_title(attr+' bar chart')
    ax.set_ylabel('number')
    ax.set_xlabel(attr)
    plt.xticks(rotation=90)

# 三种标称属性
platform = nominal_attr_statistic(data, 'Platform')
for k, v in platform.items():
    print(str(k) + ' ' + str(v))
nominal_barchart(platform, 'Platform')

genre = nominal_attr_statistic(data, 'Genre')
for k, v in genre.items():
    print(str(k) + ' ' + str(v))
nominal_barchart(genre, 'genre')

publisher = nominal_attr_statistic(data, 'Publisher')
for k, v in publisher.items():
    print(str(k) + ' ' + str(v))
nominal_barchart(publisher, 'publisher')

# 五个数值属性
na_sales = data['NA_Sales'].describe()
print(na_sales)
data['NA_Sales'].plot.box()

eu_sales = data['EU_Sales'].describe()
print(eu_sales)
data['EU_Sales'].plot.box()

jp_sales = data['JP_Sales'].describe()
print(jp_sales)
data['JP_Sales'].plot.box()

other_sales = data['Other_Sales'].describe()
print(other_sales)
data['Other_Sales'].plot.box()

global_sales = data['Global_Sales'].describe()
print(global_sales)
data['Global_Sales'].plot.box()

# 销售量随时间变化
year2sales = {}
for i in range(len(data)):
    if int(data.loc[i, 'Year']) in year2sales:
        year2sales[int(data.loc[i, 'Year'])][0] += data.loc[i, 'NA_Sales']
        year2sales[int(data.loc[i, 'Year'])][1] += data.loc[i, 'EU_Sales']
        year2sales[int(data.loc[i, 'Year'])][2] += data.loc[i, 'JP_Sales']
        year2sales[int(data.loc[i, 'Year'])][3] += data.loc[i, 'Other_Sales']
        year2sales[int(data.loc[i, 'Year'])][4] += data.loc[i, 'Global_Sales']
    else:
        year2sales[int(data.loc[i, 'Year'])] = [data.loc[i, 'NA_Sales'], data.loc[i, 'EU_Sales'], data.loc[i, 'JP_Sales'], data.loc[i, 'Other_Sales'], data.loc[i, 'Global_Sales']]
sales_pd = pd.DataFrame(year2sales.values(), index = year2sales.keys(), columns=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'])
sales_pd.sort_index(inplace = True)
print(sales_pd)
sales_pd.cumsum()
plt.figure()
sales_pd.plot()

# 平台、类型、发行商随时间的变化
year2sales.clear()
for i in range(len(data)):
    if data.loc[i, 'Platform'] == 'PS2':
        if int(data.loc[i, 'Year']) in year2sales:
            year2sales[int(data.loc[i, 'Year'])] += 1
        else:
            year2sales[int(data.loc[i, 'Year'])] = 1
platform_pd = pd.DataFrame(year2sales.values(), index = year2sales.keys(), columns = ['Platform--PS2'])
platform_pd.sort_index(inplace = True)
print(platform_pd)
platform_pd.cumsum()
plt.figure()
platform_pd.plot()

year2sales.clear()
for i in range(len(data)):
    if data.loc[i, 'Genre'] == 'Action':
        if int(data.loc[i, 'Year']) in year2sales:
            year2sales[int(data.loc[i, 'Year'])] += 1
        else:
            year2sales[int(data.loc[i, 'Year'])] = 1
genre_pd = pd.DataFrame(year2sales.values(), index = year2sales.keys(), columns = ['Genre--Action'])
genre_pd.sort_index(inplace = True)
print(genre_pd)
genre_pd.cumsum()
plt.figure()
genre_pd.plot()

year2sales.clear()
for i in range(len(data)):
    if data.loc[i, 'Publisher'] == 'Namco Bandai Games':
        if int(data.loc[i, 'Year']) in year2sales:
            year2sales[int(data.loc[i, 'Year'])] += 1
        else:
            year2sales[int(data.loc[i, 'Year'])] = 1
publisher_pd = pd.DataFrame(year2sales.values(), index = year2sales.keys(), columns = ['Genre--Action'])
publisher_pd.sort_index(inplace = True)
print(publisher_pd)
publisher_pd.cumsum()
plt.figure()
publisher_pd.plot()

# 销量最高的类型，平台，发行商
def most_sale(data, attr):
    res = {}
    year_attr_sale = {}
    for i in range(len(data)):
        year = int(data.loc[i, 'Year'])
        val = data.loc[i, attr]
        if year in year_attr_sale:
            if val in year_attr_sale[year]:
                year_attr_sale[year][val] += 1
            else:
                year_attr_sale[year][val] = 1
        else:
            year_attr_sale[year] = {}
            year_attr_sale[year][val] = 1
    for k, v in year_attr_sale.items():
        t = max(v, key = v.get)
        res[k] = t
    return res

most = most_sale(data, 'Platform')
for i in sorted(most):
    print(i, most[i])

# 提取数据
n = len(data) * 0.8
m = len(data) * 0.9
train_feature = data.loc[:n, ['Platform', 'Genre', 'Publisher']]
train_label = data.loc[:n, 'Global_Sales']
eval_feature = data.loc[n : m, ['Platform', 'Genre', 'Publisher']]
eval_label = data.loc[n : m, 'Global_Sales']
test_feature = data.loc[m:, ['Platform', 'Genre', 'Publisher']]
test_label = data.loc[m:, 'Global_Sales']
# 训练模型
model = CatBoostRegressor(iterations = 100, learning_rate = 0.05, task_type = 'GPU', loss_function = 'RMSE')
cat_features = list(range(0, 3))
model.fit(train_feature, y = train_label, cat_features = cat_features, eval_set=        (eval_feature, eval_label), verbose = 10)
# 测试模型
predict = model.predict(test_feature, prediction_type = 'RawFormulaVal', ntree_start=0, ntree_end=model.get_best_iteration())
print(predict)
# 计算RMSE
test_label.reset_index(drop = True, inplace = True)
total = 0
for i in range(len(predict)):
    total += (predict[i] - test_label[i]) * (predict[i] - test_label[i])
print('RMSE = ', math.sqrt(total/len(predict)))

# 每10条数据中，前8条作为训练集，第9条作为验证集，第10条作为测试集
train_feature = pd.DataFrame()
train_label = []
eval_feature = pd.DataFrame()
eval_label = []
test_feature = pd.DataFrame()
test_label = []

for i in range(len(data)):
    if i % 10 < 8: 
        train_feature = train_feature.append(data.loc[i, ['Platform', 'Genre', 'Publisher']])
        train_label.append(data.loc[i, 'Global_Sales'])
    elif i % 10 == 8:
        eval_feature = eval_feature.append(data.loc[i, ['Platform', 'Genre', 'Publisher']])
        eval_label.append(data.loc[i, 'Global_Sales'])
    else:
        test_feature = test_feature.append(data.loc[i, ['Platform', 'Genre', 'Publisher']])
        test_label.append(data.loc[i, 'Global_Sales'])
# 训练模型
model = CatBoostRegressor(iterations = 100, learning_rate = 0.05, task_type = 'GPU', loss_function = 'RMSE')
cat_features = list(range(0, 3))
model.fit(train_feature, y = train_label, cat_features = cat_features, eval_set=(eval_feature, eval_label), verbose = 10)
# 测试模型
predict = model.predict(test_feature, prediction_type = 'RawFormulaVal', ntree_start = 0, ntree_end = model.get_best_iteration())
# 计算RMSE
test_label.reset_index(drop = True, inplace = True)
total = 0
for i in range(len(predict)):
   total += (predict[i] - test_label[i]) * (predict[i] - test_label[i])
print('RMSE = ', math.sqrt(total/len(predict)))