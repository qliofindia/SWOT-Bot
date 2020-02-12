# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from flask import Flask, request, jsonify, redirect, url_for, flash, Response
from flask_restful import Api, Resource
from webargs import fields, validate
from webargs.flaskparser import use_args, use_kwargs, parser, abort

import datetime as dt
import numpy as np
import ktrain as kt
import json
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
api = Api(app)

class Predict(Resource):
    """An prediction endpoint."""
    Predict_args = {"data": fields.Str(required=True)}
    
    @use_kwargs(Predict_args)
    def post(Self,data):
        jdata = request.get_json()
        data=json.dumps(jdata)
        print("inside post")
        print(data)
        results=predictor.predict(data)
        class_prob=predictor.predict(data,return_proba=True).max()*100
        class_text=predictor.predict(data)
        print ('{0} - {1}%'.format(class_text,class_prob))
        json_output=[class_text,class_prob]
        return jsonify(json_output)


@parser.error_handler
def handle_request_parsing_error(err, req, schema, error_status_code, error_headers):
    """webargs error handler that uses Flask-RESTful's abort function to return
    a JSON error response to the client.
    """
    abort(error_status_code, errors=err.messages)


if __name__ == '__main__':
    model_folder = r'C:\Test\SWOT\Training\BERT\Saved_Model\saved_model_25k'
    predictor = kt.load_predictor(model_folder)
    api.add_resource(Predict, '/predict', endpoint="predict")
    print(dt.datetime.now(), 'started..')
    app_server = WSGIServer(('127.0.0.1', 8088), app)
    app_server.serve_forever()


