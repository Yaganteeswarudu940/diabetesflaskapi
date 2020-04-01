import requests
import json
url ="http://localhost:5000/diabetesprediction"
data = [[6,148,72,35,0,33.6,0.627,50]]
#data = [[14.34, 1.68, 2.7, 25.0, 98.0, 2.8, 1.31, 0.53, 2.7, 13.0, 0.57, 1.96, 660.0]]
j_data=json.dumps(data)
headers={'content-type':'application/json','Accept-Charset':'UTF-8'}
r=requests.post(url,data=j_data,headers=headers)
print(r,"Diabitic status of patient is",r.text)
