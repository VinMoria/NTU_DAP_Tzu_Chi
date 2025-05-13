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
import matplotlib.pyplot as plt
from skopt import BayesSearchCV


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

# ========== 改良版KNN 回归 ==========
param_grid_knn = {
    "n_neighbors": list(range(1, 31)),  # 不同的邻居数量
    "metric": ["euclidean", "manhattan", "chebyshev", "cosine"],  # 不同的距离度量
}

knn = KNeighborsRegressor()
grid_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring="r2")
grid_knn.fit(X_train1, y_train1)

best_knn = grid_knn.best_estimator_
y_pred_knn = best_knn.predict(X_test1) + global_mean
knn_r2 = r2_score(y_true1, y_pred_knn)
print(f"KNN R^2: {knn_r2:.4f}")

# 保存模型到文件中
with open("./models/knn.pkl", "wb") as file:
    pickle.dump(best_knn, file)

# ========== Random Forest 回归 ==========
param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_leaf": [1, 2, 5],
}

rf = RandomForestRegressor(random_state=42)
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring="r2")
grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test) + global_mean
rf_r2 = r2_score(y_true, y_pred_rf)
print(f"Random Forest R^2: {rf_r2:.4f}")

# 保存模型到文件中
with open("./models/rf.pkl", "wb") as file:
    pickle.dump(best_rf, file)

# ========== CART 决策树 回归 ==========
dt = DecisionTreeRegressor(random_state=42)
path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

param_grid_dt = {
    "max_depth": [3, 5, 8, 12, None],
    "min_samples_split": [5, 10, 20],
    "min_samples_leaf": [2, 5, 10],
    "ccp_alpha": [0.0, 0.01, 0.1, 0.2],
}

# 定义参数网格
# param_grid_dt = {
#     'max_depth': [3, 5, 8, 12,None],               # 限制树深度，防止过拟合
#     'min_samples_split': [5, 10, 20],         # 样本不多，建议设高点
#     'min_samples_leaf': [2, 5, 10],           # 每片叶子至少有几个样本
#     'ccp_alpha': ccp_alphas
# }

# param_grid_dt = {
#     'max_depth': [None, 5, 10, 15],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'ccp_alpha': ccp_alphas  # 加入ccp_alpha进行搜索
# }

grid_dt = GridSearchCV(dt, param_grid_dt, cv=5, scoring="r2")
grid_dt.fit(X_train, y_train)

best_dt = grid_dt.best_estimator_
y_pred_dt = best_dt.predict(X_test) + global_mean
dt_r2 = r2_score(y_true, y_pred_dt)
print(f"CART Decision Tree R^2: {dt_r2:.4f}")

# 保存模型到文件中
with open("./models/cart.pkl", "wb") as file:
    pickle.dump(best_dt, file)


# ========== XGBoost 回归 ==========
from skopt import BayesSearchCV

# ... rest of the import and initial code ...

def optimize_xgboost(X_train, y_train):
    search_space = {
        'n_estimators': (50, 300),  # Smaller range for faster optimization
        'max_depth': (3, 15),
        'learning_rate': (0.01, 0.3, 'log-uniform'),
        'subsample': (0.5, 1.0, 'uniform'),
        'colsample_bytree': (0.5, 1.0, 'uniform'),
        'gamma': (0, 5),
        'reg_alpha': (0, 5),
        'reg_lambda': (0, 5)
    }

    xgbr = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    opt = BayesSearchCV(
        estimator=xgbr,
        search_spaces=search_space,
        n_iter=32,
        cv=5,
        scoring='r2',
        random_state=42
    )
    
    opt.fit(X_train, y_train)
    return opt

# Call optimization function
optimized_xgb = optimize_xgboost(X_train1, y_train1)
best_xgb = optimized_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test1) + global_mean
xgb_r2 = r2_score(y_true1, y_pred_xgb)
print(f"Optimized XGBoost R^2: {xgb_r2:.4f}")

# Save the optimized model
with open("./models/optimized_xgb.pkl", "wb") as file:
    pickle.dump(best_xgb, file)

# ========== 支持向量机(SVM) 回归 ==========

param_grid_svm = {
    "C": [0.1, 1, 10],
    "epsilon": [0.01, 0.1, 0.2],
    "kernel": ["linear", "rbf"],
}

svm = SVR()
grid_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring="r2")
grid_svm.fit(X_train1, y_train1)

best_svm = grid_svm.best_estimator_
y_pred_svm = best_svm.predict(X_test1) + global_mean
svm_r2 = r2_score(y_true1, y_pred_svm)
print(f"SVM R^2: {svm_r2:.4f}")

# # 计算相对误差率
# relative_errors_svm = y_true1 - y_pred_svm

# # 绘制相对误差率图
# plt.figure(figsize=(10, 6))
# plt.bar(range(len(y_true1)), relative_errors_svm, color="skyblue")
# plt.xlabel("index")
# plt.ylabel("relative error radio")
# plt.title("SVM relative error")
# plt.show()

# 保存模型到文件中
with open("./models/svm.pkl", "wb") as file:
    pickle.dump(best_svm, file)



def plot_predictions_vs_actual_xgb(actual, predicted, title="XGBoost Prediction vs Actual"):
    plt.figure(figsize=(10, 6))
    plt.scatter(actual, predicted, color='lightcoral', edgecolor='k', alpha=0.7)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()

plot_predictions_vs_actual_xgb(y_true1, y_pred_xgb, title="XGBoost Prediction vs Actual")


def residual_plot_xgb(actual, predicted, title="XGBoost Residual Plot"):
    residuals = actual - predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(predicted, residuals, color='mediumseagreen', edgecolor='k', alpha=0.7)
    plt.hlines(y=0, xmin=predicted.min(), xmax=predicted.max(), linestyles='dashed')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.show()

residual_plot_xgb(y_true1, y_pred_xgb, title="XGBoost Residual Plot")



def actual_vs_predicted_line_plot_xgb(actual, predicted, title="XGBoost Actual vs Predicted Line Plot"):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual', color='blue')
    plt.plot(predicted, label='Predicted', color='red', linestyle='dashed')
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.title(title)
    plt.legend()
    plt.show()

actual_vs_predicted_line_plot_xgb(y_true1, y_pred_xgb, title="XGBoost Actual vs Predicted Line Plot")