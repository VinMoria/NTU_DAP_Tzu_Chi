from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
from SIBOR import cal_rate
import db_process
import json
import functions
import joblib

app = Flask(__name__)

income_cols = [
    "income_assessment_salary",
    "income_assessment_cpf_payout",
    "income_assessment_assistance_from_other_agencies",
    "income_assessment_assistance_from_relatives_friends",
    "income_assessment_insurance_payout",
    "income_assessment_rental_income",
    "income_assessment_others_income",
]

expenditure_cols = [
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
]


MODEL_PATH = "./models/svm.pkl"


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/form", methods=["GET", "POST"])
def form():
    return render_template("form.html")


@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    return render_template("feedback.html")


@app.route("/submit", methods=["POST"])
def submit_data():
    try:
        # 从请求中获取 JSON 数据
        data = request.get_json()
        df = pd.DataFrame([data])
        df_raw = df.copy(deep=True)

        # 转移日期变量
        assessment_date = df.loc[0, "assessment_date"]
        df.drop(columns=["assessment_date"], inplace=True)

        def convert_if_possible(col):
            try:
                return pd.to_numeric(col, errors="raise")  # 如果不能转为数字会抛出异常
            except ValueError:
                return col.astype("category")  # 不能转换时返回原始列

        df = df.apply(
            lambda col: convert_if_possible(col)
        )  # 只对可转换的列进行数字转换

        # 计算统计量
        df["income_total_cal"] = df[income_cols].sum(axis=1)
        df["expenditure_total_cal"] = df[expenditure_cols].sum(axis=1)
        df["difference_cal"] = df["income_total_cal"] - df["expenditure_total_cal"]

        # SIBOR调整
        if df["income_rate"][0] == -1:
            # 使用默认SIBOR
            target_date = pd.to_datetime(assessment_date)
            source_date = pd.to_datetime("2024-01-01")

            adjust_rate = cal_rate(source_date, target_date)
        else:
            # 使用用户指定值
            adjust_rate = float(df["income_rate"])

        print(adjust_rate)
        for col in income_cols:
            df[col] /= adjust_rate

        # TODO 模型预测

        # df["amount_total"] = 0
        df = functions.get_feature_columns(df, "onehot")
        print(">> get_feature_columns")
        print(df.columns)
        occ_cols = [
            "occ_employed",
            "occ_part-time",
            "occ_retired",
            "occ_unemployed",
            "occ_antenatal_check_fees_assistance",
            "occ_day_care_fees_assistance",
            "occ_education_fees_assistance",
            "occ_hiv_medication_fees",
            "occ_household_living_assistance",
            "occ_interim_dialysis_assistance",
            "occ_medical_transport_assistance",
            "occ_others",
            "occ_student",
            "occ_missing",
            "occ_medical_consumables_assistance",
        ]

        for col in occ_cols:
            if col not in df.columns:
                df[col] = 0

        df = functions.X_log(df)
        print(">> X log")

        df = functions.X_standard(df, "onehot", "yes")
        print(">> X standard")

        X = functions.xy(df, "onehot", "yes", "yes", "yes")
        print(X)
        print(">> xy")

        print(X)

        r = model_cal(X)

        df_raw["amount_total"] = r

        # 导入数据库

        profile_id = db_process.insert_case_data(df_raw)

        # 返回成功响应
        return (
            jsonify({"message": "Success", "r": str(r), "profile_id": str(profile_id)}),
            200,
        )
    except Exception as e:
        print(e)
        return jsonify({"message": "Data processing failed", "error": str(e)}), 500


@app.route("/result", methods=["GET", "POST"])
def result():
    r = request.args.get("r")
    profile_id = request.args.get("profile_id")
    return render_template("result.html", r=r, profile_id=profile_id)


@app.route("/search_profile", methods=["POST"])
def search_profile():
    try:
        # 从请求中获取 JSON 数据
        data = request.get_json()
        profile_id = int(data.get("profile_id"))

        data_dict = db_process.select_case_by_id(profile_id)
        data_dict["message"] = "Success"
        print(data_dict)
        return app.response_class(
            response=json.dumps(data_dict, ensure_ascii=False, sort_keys=False),
            status=200,
            mimetype="application/json",
        )
    except Exception as e:
        print(e)
        return jsonify({"message": "Data processing failed", "error": str(e)}), 500


@app.route("/update_feedback", methods=["POST"])
def supdate_feedback():
    try:
        # 从请求中获取 JSON 数据
        data = request.get_json()
        feedback_val = int(data.get("feedback"))
        profile_id = int(data.get("profile_id"))

        db_process.update_feedback(profile_id, feedback_val)

        # 返回成功响应
        return jsonify({"success": True})

    except Exception as e:
        # 记录错误并返回失败响应
        print(f"Error updating feedback: {e}")
        return jsonify({"success": False}), 500


def model_cal(df):
    df_history = pd.read_csv("Cleaned_Data_0506.csv")
    model = pickle.load(open(MODEL_PATH, "rb"))
    prediction = model.predict(df) + functions.return_by_global_mean(df_history)
    return prediction[0]


if __name__ == "__main__":
    app.run(debug=True)
