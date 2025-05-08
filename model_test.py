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
import xgboost as xgb
import pickle

# 读取数据
df = pd.read_csv("./data/Cleaned_Data_0506.csv")
df_raw = pd.read_csv("./data/Cleaned_Data_0506.csv")

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


global_mean = return_by_global_mean(df)
df = get_feature_columns(df, "onehot")
df_1 = X_log(df)
df_1 = X_standard(df_1, "onehot", "yes")
X1, y1 = xy(df_1, "onehot", "yes", "no", "no")

# 划分训练集和测试集
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X1, y1, test_size=0.3, random_state=42
)
y_true1 = df.loc[y_test1.index, "amount_total"]

# ========== 支持向量机(SVM) 回归 ==========

# param_grid_svm = {
#     "C": [0.1, 1, 10],
#     "epsilon": [0.01, 0.1, 0.2],
#     "kernel": ["linear", "rbf"],
# }

# svm = SVR()
# grid_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring="r2")
# grid_svm.fit(X_train1, y_train1)

# best_svm = grid_svm.best_estimator_
# y_pred_svm = best_svm.predict(X_test1) + global_mean
# svm_r2 = r2_score(y_true1, y_pred_svm)
# print(f"SVM R^2: {svm_r2:.4f}")

svm = pickle.load(open("./models/svm.pkl", "rb"))


import numpy as np  # 确保已导入NumPy库

# 从测试集中随机抽取一条数据进行预测
random_index = np.random.choice(X_test1.shape[0], size=1)
random_sample = X_test1.iloc[random_index]
random_sample_true_amount = y_true1.iloc[random_index].values[0]
random_sample_index = y_true1.index[random_index]

# 打印 random_sample 对应的原始数据
original_data_of_random_sample = df.loc[random_sample_index]
df_raw.loc[random_sample_index].to_csv("raw_sample.csv")
print(f"Original Data of Random Sample:\n{original_data_of_random_sample}")

# print(random_sample)
random_sample.to_csv("data_before_model.csv")
predicted_amount = svm.predict(random_sample) + global_mean
print(f"Random Sample True Amount: {random_sample_true_amount:.4f}")
print(f"Random Sample Predicted Amount: {predicted_amount[0]:.4f}")

