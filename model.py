from sklearn.linear_model import LogisticRegression
import pickle

def unpickle():
    path = './model.pkl'
    model_unpickle = open(path, 'rb')
    return pickle.load(model_unpickle)
