# from dataclasses import fields (unused import, can be removed)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import itertools
import numpy as np
import re

def prepare_features(df):
    df = df.copy()

    # ============ Layout: extract valid format like '3室2厅' ============
    chinese2digit = {
        '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
        '六': '6', '七': '7', '八': '8', '九': '9', '十': '10', '两': '2'
    }
    def chinese_to_digit(text):
        for ch, num in chinese2digit.items():
            text = str(text).replace(ch, num)
        return text

    df['户型'] = df['户型'].astype(str).apply(chinese_to_digit)
    pattern = re.compile(r'^\d+室\d+厅$')
    df = df[df['户型'].apply(lambda x: bool(pattern.match(x)))].copy()
    df['几室'] = df['户型'].str.extract(r'(\d+)室').astype(int)
    df['几厅'] = df['户型'].str.extract(r'(\d+)厅').astype(int)

    # ============ Area and Unit Price ============
    df['建筑面积'] = df['建筑面积'].astype(str).str.replace('㎡', '', regex=False)
    df['建筑面积'] = pd.to_numeric(df['建筑面积'], errors='coerce').astype(float)
    df['单价'] = df['单价'].astype(str).str.replace('元/㎡', '', regex=False)
    df['单价'] = pd.to_numeric(df['单价'], errors='coerce').astype(float)
    df = df.dropna(subset=['建筑面积', '单价'])

    # ============ Orientation: simplify to South or North ============
    def simplify_orientation(text):
        text = str(text)
        if '南' in text:
            return '南'
        elif '北' in text:
            return '北'
        else:
            return None
    df['主朝向'] = df['朝向'].apply(simplify_orientation)
    df = pd.get_dummies(df, columns=['主朝向'], prefix='朝向')
    if '朝向_南' in df.columns and '朝向_北' in df.columns:
        df = df[df['朝向_南'] + df['朝向_北'] == 1]
        df[['朝向_南', '朝向_北']] = df[['朝向_南', '朝向_北']].astype(int)

    # ============ Floor Level Classification ============
    def classify_floor(text):
        text = str(text)
        if '底' in text or '低' in text:
            return '低层'
        elif '中' in text:
            return '中层'
        elif '高' in text or '顶' in text:
            return '高层'
        else:
            return None
    df['楼层类型'] = df['楼层'].apply(classify_floor)
    df = df.dropna(subset=['楼层类型'])
    df = pd.get_dummies(df, columns=['楼层类型'], prefix='楼层', drop_first=True)
    df[[col for col in df.columns if col.startswith('楼层_')]] = df[[col for col in df.columns if col.startswith('楼层_')]].astype(int)

    # ============ Total Floor: extract numbers ============
    def extract_total_floor(text):
        match = re.search(r'\d+', str(text))
        return float(match.group()) if match else np.nan
    df['总楼层'] = df['总楼层'].apply(extract_total_floor)
    df['总楼层'] = df['总楼层'].fillna(df['总楼层'].median())

    # ============ Decoration Classification ============
    def classify_deco(text):
        text = str(text)
        if re.search(r'\d+', text) or '精' in text or '豪' in text:
            return '精装修'
        elif '装' in text and not ('未' in text or '无' in text):
            return '简装修'
        else:
            return '毛坯'
    df['装修分类'] = df['装修'].apply(classify_deco)
    df = pd.get_dummies(df, columns=['装修分类'], prefix='装修', drop_first=True)
    df[[col for col in df.columns if col.startswith('装修_')]] = df[[col for col in df.columns if col.startswith('装修_')]].astype(int)

    # ============ Transportation: metro proximity ============
    def classify_transport(text):
        if pd.isna(text):
            return '不近地铁'
        text = str(text).strip()
        if text != '' and '未' not in text and '无' not in text:
            return '近地铁'
        else:
            return '不近地铁'
    df['交通分类'] = df['交通'].apply(classify_transport)
    df = pd.get_dummies(df, columns=['交通分类'], prefix='交通', drop_first=True)
    df[[col for col in df.columns if col.startswith('交通_')]] = df[[col for col in df.columns if col.startswith('交通_')]].astype(int)

    return df

# Load and process data
df = pd.read_excel('../crawl/final housing information after ai.xlsx')
df_clean = prepare_features(df)

# Columns for model
cols_to_print = ['小区', '建筑面积', '单价', '总楼层', '几室', '几厅']
cols_to_print += [col for col in df_clean.columns if col.startswith('朝向_')]
cols_to_print += [col for col in df_clean.columns if col.startswith('楼层_')]
cols_to_print += [col for col in df_clean.columns if col.startswith('装修_')]
cols_to_print += [col for col in df_clean.columns if col.startswith('交通_')]

# Create target variable
df_clean['总价'] = (df_clean['建筑面积'] * df_clean['单价']).astype(float)

# Filter records with area ≤ 300
df1 = df_clean[df_clean["建筑面积"] <= 300]

# Build full model
cols = ["建筑面积", "几室", "几厅", "总楼层", "朝向_北", "朝向_南", "楼层_低层", "楼层_高层", "装修_简装修", "装修_精装修", "交通_近地铁"]
X = df1[cols]
Y = df1["总价"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

# Linear regression
linear_multi = LinearRegression()
model = linear_multi.fit(x_train, y_train)
print(model.intercept_, model.coef_)

# R² score
perdict_result = model.predict(x_test)
score = model.score(x_test, y_test)
print('R-scores:', score)

# Hypothesis testing (OLS regression for feature significance)
X = df1[cols]
Y = df1[["总价"]].values
X_ = sm.add_constant(X)
result = sm.OLS(Y, X_).fit()
print(result.summary())

# AIC-based feature selection
fields = cols
acis = {}
for i in range(1, len(fields)+1):
    for vars_combo in itertools.combinations(fields, i):
        x1 = sm.add_constant(df1[list(vars_combo)])
        res = sm.OLS(Y, x1).fit()
        acis[vars_combo] = res.aic

# Display top feature sets by lowest AIC
from collections import Counter
counter = Counter(acis)
print(counter.most_common()[::-10])

# Train model with best AIC combo
cols2 = ['建筑面积', '几厅', '总楼层', '装修_精装修', '交通_近地铁']
X = df1[cols2]
Y = df1["总价"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
model1 = LinearRegression().fit(x_train, y_train)

print(model1.intercept_, model1.coef_)
print('R-scores:', model1.score(x_test, y_test))
print('R-squares:', r2_score(model1.predict(x_test), y_test))
print('MAE:', mean_absolute_error(model1.predict(x_test), y_test))
print('MSE:', mean_squared_error(model1.predict(x_test), y_test))

# Plot prediction vs actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, model1.predict(x_test), alpha=0.6, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Total Price (¥)")
plt.ylabel("Predicted Total Price (¥)")
plt.title("Predicted vs Actual Total Price")
plt.grid(True)
plt.tight_layout()
plt.show()