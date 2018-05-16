from flask import Flask
from app.config import Config
from sklearn.externals import joblib

app = Flask(__name__)
app.config.from_object(Config)

#open model
clf = joblib.load('model.pkl')
tf = joblib.load('vektor.pkl')  

from app import routes