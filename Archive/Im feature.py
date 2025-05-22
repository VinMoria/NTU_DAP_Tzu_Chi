# 用户重新上传了数据，我马上重新执行绘制剪枝CART回归树和输出Random Forest重要特征

# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score

# 读取数据
df = pd.read_csv("Cleaned_Data__Updated_Case_Profile_Count_.csv")

# 删除目标变量中为NaN的行
df.dropna(subset=['Assistance Amount'], inplace=True)

# 简单预处理
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

df[num_cols] = num_imputer.fit_transform(df[num_cols])
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# Label Encoding
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 特征与目标
X = df.drop(["Assistance Amount", "Recommendation"], axis=1)
y = df['Assistance Amount']

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 训练Random Forest
param_grid_rf = {
    'n_estimators': [100],
    'max_depth': [5, 10, 15, None],
    'min_samples_leaf': [1, 2, 5]
}

rf = RandomForestRegressor(random_state=42)
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='r2')
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

# 训练剪枝版CART决策树
dt = DecisionTreeRegressor(random_state=42)
path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# 定义参数网格
param_grid_dt = {
    'max_depth': [3, 5, 8, 12,None],               # 限制树深度，防止过拟合
    'min_samples_split': [5, 10, 20],         # 样本不多，建议设高点
    'min_samples_leaf': [2, 5, 10],           # 每片叶子至少有几个样本
    'ccp_alpha': ccp_alphas  
}

# param_grid_dt = {
#     'max_depth': [None, 5, 10, 15],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'ccp_alpha': ccp_alphas
# }

grid_dt = GridSearchCV(dt, param_grid_dt, cv=5, scoring='r2')
grid_dt.fit(X_train, y_train)
best_dt = grid_dt.best_estimator_

# ========== 绘制剪枝后的CART回归树（只画前3层） ==========
plt.figure(figsize=(20, 10))
plot_tree(best_dt,
          filled=True,
          feature_names=X.columns,
          rounded=True,
          max_depth=3)  # 只展示前3层
plt.title("Pruned CART Decision Tree (First 3 Levels)")
plt.show()

# ========== 输出Random Forest特征重要性 ==========
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nTop 15 Feature Importances from Random Forest (Regression):")
for i in range(15):
    print(f"{i+1}. {X.columns[indices[i]]}: {importances[indices[i]]:.4f}")



# 绘制特征重要性图
plt.figure(figsize=(12, 6))
plt.bar(range(15), importances[indices[:15]], align="center")
plt.xticks(range(15), X.columns[indices[:15]], rotation=45, ha="right")
plt.title("Top 15 Important Features (Random Forest - Regression)")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()
