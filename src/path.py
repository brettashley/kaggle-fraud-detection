from flask import Flask, render_template, jsonify
app = Flask(__name__, static_url_path='')
from model import unpickle
from sklearn.linear_model import LogisticRegression
import pickle
import random


@app.route('/hello')
def hello_world():
    return 'Hello, World!'

@app.route('/results')
def return_results():
    model = unpickle()
    probs = model.predict_proba([[135977.22,      0.  ,      0.  ]])
    return render_template('result.html', result=str(probs))

@app.route('/json')
def pred_json():
    model = unpickle()
    probs = model.predict_proba([[135977.22,      0.  ,      0.  ]])
    return jsonify({'proba': str(probs)})



@app.route('/')
def index():
    """Return the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Return a random prediction."""
    return jsonify({'probability': random.random()})
