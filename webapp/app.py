from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

MODEL_PATH = './models/xgb.pkl'

@app.route( '/', methods=['GET','POST'])
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit_data():
    try:
        # 从请求中获取 JSON 数据
        data = request.get_json()
        df = pd.DataFrame([data])

        def convert_if_possible(col):
            try:
                return pd.to_numeric(col, errors='raise')  # 如果不能转为数字会抛出异常
            except ValueError:
                return col.astype('category')  # 不能转换时返回原始列
        
        df = df.apply(lambda col: convert_if_possible(col))  # 只对可转换的列进行数字转换
        print(df.dtypes)
        print(df)
        r = model_cal(df)
        # 返回成功响应
        return jsonify({"message": "Success", "r": str(r)}), 200
    except Exception as e:
        print(e)
        return jsonify({"message": "Data processing failed", "error": str(e)}), 500


@app.route('/result', methods=['GET','POST'])
def result():
    r = request.args.get('r')
    return render_template('result.html', r=r)

def model_cal(df):
    model = pickle.load(open(MODEL_PATH, 'rb'))
    prediction = model.predict(df)
    # 修改为预测的第一个元素
    return prediction[0]


if __name__ == "__main__":
    app.run(debug=True)
