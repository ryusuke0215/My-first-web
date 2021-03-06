from flask import Flask, render_template, request
from flask import redirect, url_for
import pickle
import json
import numpy as np

app          = Flask(__name__)
iris         = ["Setosa", "Versicolor", "Virginica"]
datasets     = {'data': [], 'answer': []}
sepal_length = None
sepal_width  = None
petal_length = None
petal_width  = None
count        = 0

# オブジェクトの呼び出し
with open('./app/static/model/model.pickle', 'rb') as f:
    clf = pickle.load(f)

# index.htmlへ移動
@app.route("/")
def index():
    return render_template("index.html")

# 予測
@app.route("/pred", methods=["post"])
def pred():
    global iris, clf, sepal_length, sepal_width, petal_length, petal_width
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

# add.htmlへ移動
@app.route("/add")
def add():
    global iris, sepal_length, sepal_width, petal_length, petal_width
    return render_template("add.html", sepal_length=sepal_length, sepal_width=sepal_width, petal_length=petal_length, petal_width=petal_width, iris=iris)

# データを追加
@app.route("/add_data", methods=["post"])
def add_data():
    global count, datasets, iris, sepal_length, sepal_width, petal_length, petal_width
    try:
        new_sepal_length = float(request.form["sepal_length"])
        new_sepal_width  = float(request.form["sepal_width"])
        new_petal_length = float(request.form["petal_length"])
        new_petal_width  = float(request.form["petal_width"])
        new_answer       = int(iris.index(request.form["answer"]))
        
    except:
        return render_template("add.html", sepal_length=sepal_length, sepal_width=sepal_width, petal_length=petal_length, petal_width=petal_width, iris=iris, error="error")
    
    datasets["data"].append(new_sepal_length)
    datasets["data"].append(new_sepal_width)
    datasets["data"].append(new_petal_length)
    datasets["data"].append(new_petal_width)
    datasets["answer"].append(new_answer)

    sepal_length = None
    sepal_width  = None
    petal_length = None
    petal_width  = None
    count       += 1
    
    with open("./app/static/jsons/{:06d}.json".format(count), "w") as f:
        json.dump(datasets, f, indent=4, ensure_ascii=False)
    
    datasets = {'data': [], 'answer': []}
    return render_template("Top.html")

# データを見る
#@app.route("/show_data")
#def show_data():
#    global datasets
#    return render_template("show.html", datasets=datasets)

if __name__ == "__main__":
    app.run(debug=True)