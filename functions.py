import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import joblib


def get_log_columns():
    return [
        'income_assessment_salary',
        'income_assessment_cpf_payout',
        'income_assessment_assistance_from_other_agencies',
        'income_assessment_assistance_from_relatives_friends',
        'income_assessment_insurance_payout',
        'income_assessment_rental_income',
        'income_assessment_others_income',
        'expenditure_assessment_mortgage_rental',
        'expenditure_assessment_utilities',
        'expenditure_assessment_s_cc_fees',
        'expenditure_assessment_food_expenses',
        'expenditure_assessment_marketing_groceries',
        'expenditure_assessment_telecommunications',
        'expenditure_assessment_transportation',
        'expenditure_assessment_medical_expenses',
        'expenditure_assessment_education_expense',
        'expenditure_assessment_contribution_to_family_members',
        'expenditure_assessment_domestic_helper',
        'expenditure_assessment_loans_debts_installments',
        'expenditure_assessment_insurance_premiums',
        'expenditure_assessment_others_expenditure',
        'income_total_cal',
        'expenditure_total_cal',
        'difference_cal',
        'current_savings'
    ]

def get_other_columns(method):
    if method == "onehot":
        return [
            'income_ratio_filled', 'income_ratio_missing',
            'primary_7_12_ratio_filled', 'primary_7_12_ratio_missing',
            'secondary_13_17_ratio_filled', 'secondary_13_17_ratio_missing',
            'tertiary_18_21_ratio_filled', 'tertiary_18_21_ratio_missing',
            'adult_22_64_ratio_filled', 'adult_22_64_ratio_missing',
            'elderly_65_and_above_ratio_filled', 'elderly_65_and_above_ratio_missing',
            'has_no_of_hh', 'age', 'occupation_missing',
            'occ_employed', 'occ_missing', 'occ_part-time', 'occ_retired',
            'occ_student', 'occ_unemployed', 'occ_antenatal_check_fees_assistance',
            'occ_day_care_fees_assistance', 'occ_education_fees_assistance',
            'occ_hiv_medication_fees', 'occ_household_living_assistance',
            'occ_interim_dialysis_assistance', 'occ_medical_consumables_assistance',
            'occ_medical_transport_assistance', 'occ_others'
        ]
    elif method == "label":
        return [
            'income_ratio_filled', 'income_ratio_missing',
            'primary_7_12_ratio_filled', 'primary_7_12_ratio_missing',
            'secondary_13_17_ratio_filled', 'secondary_13_17_ratio_missing',
            'tertiary_18_21_ratio_filled', 'tertiary_18_21_ratio_missing',
            'adult_22_64_ratio_filled', 'adult_22_64_ratio_missing',
            'elderly_65_and_above_ratio_filled', 'elderly_65_and_above_ratio_missing',
            'has_no_of_hh', 'age', 'occupation_encoded', 'occupation_missing',
            'type_of_assistances_encoded'
        ]
    else:
        raise ValueError("method must be either 'onehot' or 'label'")



# 构建新特征
def get_feature_columns(df, method):
    
    # step 1: 
    # 处理 'intake_no_of_hh', 'no_of_hh', 'before_primary', 'primary_7_12', 'secondary_13_17', 'tertiary_18_21', 'adult_22_64', 'elderly_65_and_above' 列
    # 定义缺失或为 0 的情况
    def is_missing(x):
        return (pd.isnull(x)) | (x == 0)
    
    # 标志：是否有 no_of_hh 数据
    df["has_no_of_hh"] = (~is_missing(df["no_of_hh"])).astype(int)

    # 人均收入比例
    df["income_ratio"] = df["intake_no_of_hh"] / df["no_of_hh"]
    df["income_ratio"] = df["income_ratio"].replace([np.inf, -np.inf], np.nan).fillna(-1)

    df["income_ratio_missing"] = (df["income_ratio"] == -1).astype(int)
    df["income_ratio_filled"] = df["income_ratio"]

    # 年龄段列构建比例 + 缺失信息
    age_cols = ['primary_7_12', 'secondary_13_17', 'tertiary_18_21', 'adult_22_64', 'elderly_65_and_above']
    for col in age_cols:
        df[f"{col}_ratio"] = df[col] / df["no_of_hh"]
        df[f"{col}_ratio_filled"] = df[f"{col}_ratio"].fillna(-1)
        df[f"{col}_ratio_filled"] = df[f"{col}_ratio_filled"].replace([np.inf, -np.inf], np.nan).fillna(-1)
        df[f"{col}_ratio_missing"] = df[f"{col}_ratio"].isnull().astype(int)


    # step 2:
    # 处理age

    # step 3:
    # 处理occupation
    # 缺失填补
    df["occupation_filled"] = df["occupation"].fillna("missing")

    # 缺失标志列
    df["occupation_missing"] = df["occupation"].isnull().astype(int)

    if method == "onehot":
        # One-hot 编码
        df_onehot = pd.get_dummies(df["occupation_filled"], prefix="occ")
        df = pd.concat([df, df_onehot], axis=1)
    elif method == "label":
        # Label Encoding
        le = LabelEncoder()
        df["occupation_encoded"] = le.fit_transform(df["occupation_filled"])
    else:
        raise ValueError("method must be either 'onehot' or 'label'")

    # step 4:
    # 处理 income & expenditure

    # step 5:
    # 处理 current_saving 列
    df.loc[df["current_savings"] < 0, "current_savings"] = 0
    
    # step 6:
    # 处理 type_of_assistances
    if method == "onehot":
        # One-hot 编码
        df_onehot = pd.get_dummies(df["type_of_assistances"], prefix="occ")
        df = pd.concat([df, df_onehot], axis=1)
    elif method == "label":
        # Label Encoding
        le = LabelEncoder()
        df["type_of_assistances_encoded"] = le.fit_transform(df["type_of_assistances"])
    else:
        raise ValueError("method must be either 'onehot' or 'label'")
    
    # step 7:
    # points
    return df


def X_log(df):
    # 对特征做log
    # KNN中，如果某些特征数值跨度特别大（例如：存款从 0 到 1,000,000），它们将主导距离计算，让模型忽略其他小尺度特征（如年龄、人数）
    df['difference_cal'] = - df['difference_cal']
    log_cols = get_log_columns()
     
    # 对这些列统一进行 log1p 变换
    df_log = np.log1p(df[log_cols])
    df_log = df_log.fillna(0) #有一些difference_cal是负值，log后变成空值，用0填充

    df[["log_" + col for col in log_cols]] = df_log

    return df



def X_standard(df, method, if_log):
    
    log_cols = get_log_columns()
    other_cols = get_other_columns(method)
    
    if if_log == 'yes':
        log_cols = ["log_" + col for col in log_cols]
    else:
        pass

    X_cols = log_cols + other_cols

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[X_cols])
    df[X_cols] = X_scaled  # 覆盖原有列
    joblib.dump(scaler, "scaler.pkl")
    print("scaler dump")

    return df


def xy(df, method, if_log, if_pca, if_predict):
    log_cols = get_log_columns()
    other_cols = get_other_columns(method)
    
    if if_log == 'yes':
        log_cols = ["log_" + col for col in log_cols]
    else:
        pass

    X_cols = log_cols + other_cols

    if if_pca == 'yes':
        # # PCA 降维（例如保留95%的方差）
        pca = PCA(n_components=0.95)
        X = pca.fit_transform(df[X_cols])
        # joblib.dump(pca, "pca.pkl")
        # print("pca Dump")
        # X = df[X_cols]
    else:
        X = df[X_cols]

    if if_predict == 'no':
        team_mean = df.groupby('care_team')['amount_total'].mean()
        df["team_mean"] = df['care_team'].map(team_mean)
        df["amount_adjusted_label"] = df['amount_total'] - df["team_mean"]
        y = df["amount_adjusted_label"]
        return X,y
    else:
        return X


def return_by_global_mean(df):
    team_mean = df.groupby("care_team")["amount_total"].mean()
    global_mean = team_mean.mean()
    return global_mean
