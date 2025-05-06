def X_log(df, method):
    # 对特征做log
    # KNN中，如果某些特征数值跨度特别大（例如：存款从 0 到 1,000,000），它们将主导距离计算，让模型忽略其他小尺度特征（如年龄、人数）
    df['difference_cal'] = - df['difference_cal']
    log_cols = ['income_assessment_salary',
        'income_assessment_cpf_payout',
        'income_assessment_assistance_from_other_agencies',
        'income_assessment_assistance_from_relatives_friends',
        'income_assessment_insurance_payout', 'income_assessment_rental_income',
        'income_assessment_others_income',
        'expenditure_assessment_mortgage_rental',
        'expenditure_assessment_utilities', 'expenditure_assessment_s_cc_fees',
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
        'expenditure_assessment_others_expenditure', 'income_total_cal',
        'expenditure_total_cal', 'difference_cal', 'current_savings'
        ]

    if method == "onehot":
        other_cols = ['income_ratio_filled', 'income_ratio_missing',
            'primary_7_12_ratio_filled', 'primary_7_12_ratio_missing',
            'secondary_13_17_ratio_filled', 'secondary_13_17_ratio_missing',
            'tertiary_18_21_ratio_filled', 'tertiary_18_21_ratio_missing',
            'adult_22_64_ratio_filled', 'adult_22_64_ratio_missing',
            'elderly_65_and_above_ratio_filled',
            'elderly_65_and_above_ratio_missing', 'has_no_of_hh', 'age',
            'occupation_missing', 'occ_employed', 'occ_missing', 'occ_part-time',
            'occ_retired', 'occ_student', 'occ_unemployed', 
            'occ_antenatal_check_fees_assistance', 'occ_day_care_fees_assistance',
            'occ_education_fees_assistance', 'occ_hiv_medication_fees',
            'occ_household_living_assistance', 'occ_interim_dialysis_assistance',
            'occ_medical_consumables_assistance',
            'occ_medical_transport_assistance', 'occ_others','points']
    elif method == "label":
        other_cols = ['income_ratio_filled', 'income_ratio_missing',
        'primary_7_12_ratio_filled', 'primary_7_12_ratio_missing',
        'secondary_13_17_ratio_filled', 'secondary_13_17_ratio_missing',
        'tertiary_18_21_ratio_filled', 'tertiary_18_21_ratio_missing',
        'adult_22_64_ratio_filled', 'adult_22_64_ratio_missing',
        'elderly_65_and_above_ratio_filled',
        'elderly_65_and_above_ratio_missing', 'has_no_of_hh', 'age',
        'occupation_encoded', 'occupation_missing', 
        'type_of_assistances_encoded','points']
    else:
        raise ValueError("method must be either 'onehot' or 'label'")
        
    # 对这些列统一进行 log1p 变换
    df_log = np.log1p(df[log_cols])
    df_log = df_log.fillna(0) #有一些difference_cal是负值，log后变成空值，用0填充

    df[["log_" + col for col in log_cols]] = df_log

    # 构建最终特征列（使用 log 后缀 + 其他）
    feature_cols = ["log_" + col for col in log_cols] + other_cols
    X = df[feature_cols]

    return X


def X_standard(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def X_PCA(X):
    # PCA 降维（例如保留95%的方差）
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X)
    return X_pca

def adjust_label_by_group_mean(df):
    team_mean = df.groupby('care_team')['amount_total'].mean()
    df["team_mean"] = df['care_team'].map(team_mean)
    df["amount_adjusted_label"] = df['amount_total'] - df["team_mean"]
    y_adjusted = df["amount_adjusted_label"]
    return y_adjusted

def return_by_global_mean(df):
    team_mean = df.groupby("care_team")["amount_total"].mean()
    global_mean = team_mean.mean()
    return global_mean
