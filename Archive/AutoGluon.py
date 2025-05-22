from SIBOR import cal_rate
from functions import (
    get_log_columns,
    get_other_columns,
    get_feature_columns,
    X_log,
    X_standard,
    xy,
    return_by_global_mean,
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor

def plot_feature_importances(importances, feature_names, title, n=5):
    indices = np.argsort(importances)[-n:]  # Get indices of most important features
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.barh(range(n), importances[indices], align='center')
    plt.yticks(range(n), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

# 读取数据
df = pd.read_csv("./data/Cleaned_Data_0509.csv")

Q1 = df["amount_total"].quantile(0.25)
Q3 = df["amount_total"].quantile(0.75)
IQR = Q3 - Q1

# 定义上下限：1.5倍IQR范围之外视为极端值
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 只保留非极端值
df = df[(df["amount_total"] >= lower_bound) & (df["amount_total"] <= upper_bound)]

# # 删除目标变量中为 NaN 的行
# df.dropna(subset=['amount_total'], inplace=True)

# # 简单数据预处理
# num_cols = df.select_dtypes(include=['float64', 'int64']).columns
# cat_cols = df.select_dtypes(include=['object']).columns

# num_imputer = SimpleImputer(strategy='median')
# cat_imputer = SimpleImputer(strategy='most_frequent')

# df[num_cols] = num_imputer.fit_transform(df[num_cols])
# df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# # Label Encoding 类别变量
# label_encoders = {}
# for col in cat_cols:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     label_encoders[col] = le


def calculate_adjusted_values(df, columns):
    source_date = pd.to_datetime("2024-01-01")
    df["assessment_date_time"] = pd.to_datetime(
        df["assessment_date_time"], errors="coerce"
    )  # 使用 errors='coerce'
    for column in columns:
        df[column] = df.apply(
            lambda row: (
                row[column] / cal_rate(source_date, row["assessment_date_time"])
                if pd.notna(row["assessment_date_time"])
                else row[column]
            ),
            axis=1,
        )


# ... 其余代码 ...
# 设置想要调整的列
columns_to_adjust = [
    "income_assessment_salary",
    "income_assessment_cpf_payout",
    "income_assessment_assistance_from_other_agencies",
    "income_assessment_assistance_from_relatives_friends",
    "income_assessment_insurance_payout",
    "income_assessment_rental_income",
    "income_assessment_others_income",
    "expenditure_assessment_mortgage_rental",
    "expenditure_assessment_utilities",
    "expenditure_assessment_s_cc_fees",
    "expenditure_assessment_food_expenses",
    "expenditure_assessment_marketing_groceries",
    "expenditure_assessment_telecommunications",
    "expenditure_assessment_transportation",
    "expenditure_assessment_medical_expenses",
    "expenditure_assessment_education_expense",
    "expenditure_assessment_contribution_to_family_members",
    "expenditure_assessment_domestic_helper",
    "expenditure_assessment_loans_debts_installments",
    "expenditure_assessment_insurance_premiums",
    "expenditure_assessment_others_expenditure",
    "income_total_cal",
    "expenditure_total_cal",
    "difference_cal",
    "amount_total",
]

# 使用函数调整数据
calculate_adjusted_values(df, columns_to_adjust)

# # 选择特征和目标
# X = df.drop(["amount_total", "care_team","assessment_date_time"], axis=1)
# y = df['amount_total']

# # 标准化
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

global_mean = return_by_global_mean(df)
print(global_mean)
df = get_feature_columns(df, "onehot")
df_1 = X_log(df)
df_1 = X_standard(df_1, "onehot", "yes")
X1, y1 = xy(df_1, "onehot", "yes", "no", "no")
df = X_standard(df, "onehot", "no")
X, y = xy(df, "onehot", "no", "yes", "no")

# 划分训练集和测试集
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X1, y1, test_size=0.3, random_state=22
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
y_true1 = df.loc[y_test1.index, "amount_total"]
y_true = df.loc[y_test.index, "amount_total"]

# 确保使用在转换成 numpy 数组前的 DataFrame 的列名

# 这些 DataFrames 会保留出现在 split 函数调用之前的数据列名
# 确保 'df' 是进行任何特征工程前的原始 DataFrame，或者是经过处理后保留了列名的 DataFrame

X_train_df = pd.DataFrame(X_train, columns=df.columns[:X_train.shape[1]])
y_train_df = pd.DataFrame(y_train, columns=['amount_total'])
X_test_df = pd.DataFrame(X_test, columns=df.columns[:X_test.shape[1]])
y_true_df = pd.DataFrame(y_true, columns=['amount_total'])

X_train1_df = pd.DataFrame(X_train1, columns=df_1.columns[:X_train1.shape[1]])
y_train1_df = pd.DataFrame(y_train1, columns=['amount_total'])
X_test1_df = pd.DataFrame(X_test1, columns=df_1.columns[:X_test1.shape[1]])
y_true1_df = pd.DataFrame(y_true1, columns=['amount_total'])

# 合并 DataFrames
train_data1 = pd.concat([X_train1_df, y_train1_df], axis=1)
test_data1 = pd.concat([X_test1_df, y_true1_df], axis=1)

train_data = pd.concat([X_train_df, y_train_df], axis=1)
test_data = pd.concat([X_test_df, y_true_df], axis=1)

# 确保 'amount_total' 是目标变量
label = 'amount_total'

# 使用 X1, y1 数据集进行 AutoGluon 训练
predictor1 = TabularPredictor(label=label, eval_metric='r2').fit(train_data1)
y_pred_ag1 = predictor1.predict(test_data1.drop(columns=label))
y_pred_ag1 += global_mean
ag1_r2 = r2_score(y_true1, y_pred_ag1)
print(f"AutoGluon X1, y1 R^2: {ag1_r2:.4f}")