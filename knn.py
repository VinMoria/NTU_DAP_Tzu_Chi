import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 读取数据
df = pd.read_csv("Cleaned_Data.csv")

# 删除目标变量中为 NaN 的行
df.dropna(subset=['Assistance Amount'], inplace=True)

# 特征和目标变量
X = df.drop(["Assistance Amount", "Recommendation"], axis=1)
y = df['Assistance Amount']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = []
cat_cols = []

for col in X:
    if df[col].dtypes == object:
        cat_cols.append(col)
    else:
        num_cols.append(col)

# 定义预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # 填充缺失值
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', OneHotEncoder(), cat_cols)
    ])

# 创建 KNN 回归管道
knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', KNeighborsRegressor())
])

# 定义参数网格
param_grid = {
    'regressor__n_neighbors': [3, 5, 7, 9, 11],  # 不同的邻居数量
    'regressor__metric': ['euclidean', 'manhattan', 'chebyshev', 'cosine']  # 不同的距离度量
}

# 使用 GridSearchCV 进行超参数搜索
grid_search = GridSearchCV(knn_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)

# 训练模型
grid_search.fit(X_train, y_train)

# 输出最佳参数和最佳得分
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best R^2 score: {grid_search.best_score_}')

# 使用最佳模型评估测试集
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f'Test R^2 score: {test_score}')
