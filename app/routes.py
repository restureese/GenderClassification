from app import app
from app import tf,clf
import json

import pandas as pd

from flask import render_template,url_for, request, jsonify

@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/prediksi',methods=["GET", "POST"])
def prediksi():
	if request.method == "POST":
		nama = tf.transform([str(request.form['namalengkap'])])
		predict = clf.predict(nama)[0]
		kelamin = 'Laki - Laki' if predict == 1 else 'Perempuan'
		result = 'Nama {nama} berjenis kelamin {kelamin}'.format(nama=request.form['namalengkap'],kelamin=kelamin)
		return json.dumps({"result":result})

@app.route('/tambahData',methods=["GET", "POST"])
def tambahData():
	if request.method == "POST":
		nama = str(request.form['namalengkap'])
		kelamin = str(request.form['jeniskelamin'])

		dataset = pd.read_csv('static/dataset.csv')
		data = pd.DataFrame([[nama,kelamin]],columns=['name','gender'])
		dataset.append(data,ignore_index=True)
		dataset.to_csv('static/dataset.csv')

		result = 'Nama {nama} berjenis kelamin {kelamin} berhasil ditambahkan'.format(nama=nama,kelamin=kelamin)
		return json.dumps({"result":result})