import pandas as pd


def cal_rate(source_date, target_date):
    df = pd.read_csv("SIBOR.csv")
    df["SIBOR DATE"] = pd.to_datetime(df["SIBOR DATE"], dayfirst=True)

    # 确定日期范围的正确顺序
    start_date, end_date = min(source_date, target_date), max(source_date, target_date)
    
    # 计算天数差
    day_diff = (target_date - source_date).days
    # 筛选数据
    filtered_df = df[(df["SIBOR DATE"] >= start_date) & (df["SIBOR DATE"] <= end_date)]

    # print(filtered_df)
    # 计算 SIBOR 1M 的平均值
    average_sibor_1m = filtered_df["SIBOR 1M"].mean()

    rate = (1 +(average_sibor_1m/100.0)) ** (day_diff/365.0)

    # print(f"平均值: {average_sibor_1m}, 天数差异: {day_diff}, rate: {rate}")
    return rate


if __name__ == "__main__":
    # 构建输入参数
    source_date = pd.to_datetime("2022-01-10")
    target_date = pd.to_datetime("2023-06-10")

    # 调用测试函数
    cal_rate(source_date, target_date)
