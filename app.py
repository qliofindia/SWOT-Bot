# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from flask import Flask, request, jsonify, redirect, url_for, flash, Response
import datetime as dt
import numpy as np
import ktrain as kt
import json
from gevent.pywsgi import WSGIServer
from urllib.parse import *

app = Flask(__name__)

@app.route('/api/', methods=['POST'])
def predict():
    if 'TEXT' in request.args:
        jdata = unquote(request.args['TEXT'])
        data=json.dumps(jdata)
        print(data)
        class_prob=predictor.predict(data,return_proba=True).max()*100
        class_text=predictor.predict(data)
        print ('{0} - {1}%'.format(class_text,class_prob))
        json_output=[class_text,class_prob]
        return jsonify(json_output)
    else:
        return "Error: No TEXT field provided. Please contact admin for assistance."
    

if __name__ == '__main__':
    model_folder = r'.\run264_16\saved_model_25k'
    predictor = kt.load_predictor(model_folder)
    print(dt.datetime.now(), 'started..')
    app_server = WSGIServer(('127.0.0.1', 8088), app)
    app_server.serve_forever()
    
