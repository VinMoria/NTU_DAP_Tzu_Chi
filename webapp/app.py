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

adjust_cols = [
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

        # special case label 仅用于前端显示
        hint = get_hint(df)
        # special_cases = df.loc[0, "special_cases"]
        df.drop(columns=["special_cases"], inplace=True)

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

        # print(adjust_rate)
        for col in adjust_cols:
            df[col] /= adjust_rate

        # df["amount_total"] = 0
        df = functions.get_feature_columns(df, "onehot")
        print(">> get_feature_columns")
        # print(df.columns)
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

        X = functions.xy(df, "onehot", "yes", "no", "yes")
        # print(X)
        print(">> xy")

        X.to_csv("web_data_before_model.csv")
        print("res: ", model_cal(X))
        r = round(model_cal(X) * adjust_rate, 2)

        df_raw["amount_total"] = r

        # 导入数据库

        profile_id = db_process.insert_case_data(df_raw)

        # 组装一个额外的copayment提示

        # 返回成功响应
        return (
            jsonify(
                {
                    "message": "Success",
                    "hint": hint,
                    "r": str(r),
                    "profile_id": str(profile_id),
                }
            ),
            200,
        )
    except Exception as e:
        print(e)
        return jsonify({"message": "Data processing failed", "error": str(e)}), 500


@app.route("/result", methods=["GET", "POST"])
def result():
    r = request.args.get("r")
    profile_id = request.args.get("profile_id")
    hint = request.args.get("hint")
    return render_template("result.html", r=r, profile_id=profile_id, hint=hint)


@app.route("/search_profile", methods=["POST"])
def search_profile():
    try:
        # 从请求中获取 JSON 数据
        data = request.get_json()
        profile_id = int(data.get("profile_id"))

        data_dict = db_process.select_case_by_id(profile_id)
        data_dict["message"] = "Success"
        # print(data_dict)
        return app.response_class(
            response=json.dumps(data_dict, ensure_ascii=False, sort_keys=False),
            status=200,
            mimetype="application/json",
        )
    except Exception as e:
        print(e)
        return jsonify({"message": "Data processing failed", "error": str(e)}), 500


@app.route("/update_feedback", methods=["POST"])
def update_feedback():
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
    # df_history = pd.read_csv("Cleaned_Data_0506.csv")
    model = pickle.load(open(MODEL_PATH, "rb"))
    prediction = model.predict(df) + 813.392691382825
    return prediction[0]


def get_hint(df):
    case_check_list = [
        "medical_transport_assistance",
        "medical_consumables_assistance",
        "interim_dialysis_assistance",
    ]

    apply_type = df.loc[0, "type_of_assistances"]
    case_str = df.loc[0, "special_cases"]

    res = ""

    if apply_type in case_check_list:
        apply_type = " ".join(word.capitalize() for word in apply_type.split("_"))
        res = f"The type of this case is {apply_type}. "
    if len(case_str) > 0:
        res += f"It includes specical cases: {case_str}. "

    if len(res) > 0:
        res += "May consider copayment or other customised assistance options."

    return res


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=5000)
