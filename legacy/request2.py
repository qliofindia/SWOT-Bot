# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:37:25 2020

@author: santhanaml
"""

import requests
import json

url = 'http://127.0.0.1:8088/predict'

data = ['Technology has journeyed from hardware to enterprise software to SMAC (social media, analytics and cloud) and artificial intelligence, while at the same time; it has become an integral part of every industry in an increasingly multi-device connected world. Smart machines, cognitive computing and internet of things are narrowing divide between humans and machines, and creating new applications in a world of cognizant computing where technology will take decisions for humans based on historical data and preferences. Thus, Information Technology is continuously evolving and the trend is expected to continue in future.']
para = {'data':json.dumps(data)}
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
c = requests.post(url=url, params=para, headers=headers)
print(c.text)

