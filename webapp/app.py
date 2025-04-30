from flask import Flask, request, render_template

app = Flask(__name__)



@app.route( '/', methods=['GET','POST'])
def index():
    return render_template('index.html')


@app.route("/submit', methods=['POST'])")
def submit_data():
    pass


if __name__ == "__main__":
    app.run(debug=True)