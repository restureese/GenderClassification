from flask import Flask
from app.config import Config
import joblib

app = Flask(__name__)
app.config.from_object(Config)

#open model
clf = joblib.load('app/model.pkl')
tf = joblib.load('app/vektor.pkl')  

from app import routes