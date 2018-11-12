from flask import Flask, render_template
app = Flask(__name__, static_url_path='')
from model import unpickle
from sklearn.linear_model import LogisticRegression
import pickle


@app.route('/hello')
def hello_world():
    return 'Hello, World!'

@app.route('/results')
def predict():
    model = unpickle()
    probs = model.predict_proba([[135977.22,      0.  ,      0.  ]])
    return render_template('result.html', result=str(probs))
