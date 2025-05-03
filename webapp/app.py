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
        print(df.dtypes)
        r = model_cal(df)
        # 返回成功响应
        return jsonify({"message": "Success", "r": r}), 200
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
    return prediction[0, 0]


if __name__ == "__main__":
    app.run(debug=True)
