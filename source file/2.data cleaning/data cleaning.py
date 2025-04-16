
import pandas as pd
import numpy as np
import re

def prepare_features(df):
    df = df.copy()

    # ============ Layout: extract valid floor plan info (e.g. 3室2厅) ============
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
    df['建筑面积'] = pd.to_numeric(df['建筑面积'], errors='coerce')
    df["建筑面积"] = df["建筑面积"].astype(float)
    df['单价'] = df['单价'].astype(str).str.replace('元/㎡', '', regex=False)
    df['单价'] = pd.to_numeric(df['单价'], errors='coerce')
    df["单价"] = df["单价"].astype(float)
    df = df.dropna(subset=['建筑面积', '单价'])

    # ============ Orientation simplification: only keep South or North ============
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

    # ============ Floor classification: low / mid / high ============
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

    # ============ Total floor: extract number only ============
    def extract_total_floor(text):
        match = re.search(r'\d+', str(text))
        return float(match.group()) if match else np.nan
    df['总楼层'] = df['总楼层'].apply(extract_total_floor)
    df['总楼层'] = df['总楼层'].fillna(df['总楼层'].median())

    # ============ Decoration level classification ============
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

# Load and clean data
df = pd.read_excel('../crawl/final house information after ai.xlsx')
df_clean = prepare_features(df)

# Select model-related columns
cols_to_print = ['小区', '建筑面积', '单价', '总楼层', '几室', '几厅']
cols_to_print += [col for col in df_clean.columns if col.startswith('朝向_')]
cols_to_print += [col for col in df_clean.columns if col.startswith('楼层_')]
cols_to_print += [col for col in df_clean.columns if col.startswith('装修_')]
cols_to_print += [col for col in df_clean.columns if col.startswith('交通_')]

print(df_clean[cols_to_print].head(10))
df_clean[cols_to_print].to_excel('../data cleaning/final data cleaning.xlsx', index=False)
