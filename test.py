from SIBOR import cal_rate
import pandas as pd

# 构建输入参数
source_date = pd.to_datetime("2022-01-10")
target_date = pd.to_datetime("2023-06-10")

# 调用测试函数
print( cal_rate(source_date, target_date))