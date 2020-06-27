from flask import Flask, render_template, request
from flask import redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)
iris = ["Setosa", "Versicolor", "Virginica"]

# オブジェクトの呼び出し
with open('./app/model.pickle', 'rb') as f:
    clf = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/pred", methods=["post"])
def pred():
    global iris, clf
    try:
        sepal_length = float(request.form["sepal_length"])
        sepal_width  = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width  = float(request.form["petal_width"])
    except:
        return render_template("index.html", error="error")

    input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    index = clf.predict(input)[0]
    name = iris[index]
    return render_template("index.html", name=name)

if __name__ == "__main__":
    app.run(debug=True)