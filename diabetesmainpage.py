from flask import Flask,render_template,jsonify,request
import numpy as np
import pandas as pd
import sklearn
import json
import pickle as p
import requests
app=Flask(__name__)

@app.route("/diabetesprediction", methods=['POST'])
def predictdiabetes():
    data=request.get_json()
    prediction=np.array2string(model.predict(data))
    return jsonify(prediction)

@app.route('/')
def index():
    return render_template("diabetesmainpage.html")
@app.route('/diabetescondition',methods=['POST'])
def diabetescondition():
    url="http://localhost:5000/diabetesprediction"
    Pregnancies = request.form['Pregnancies']
    Glucose = request.form['Glucose']
    BP = request.form['BP']
    ST = request.form['ST']
    Insulin = request.form['Insulin']
    BMI = request.form['BMI']
    DPF = request.form['DPF']
    Age = request.form['Age']
    data=[[Pregnancies,Glucose,BP,ST,Insulin,BMI,DPF,Age]]
    j_data=json.dumps(data)
    headers={'content-type':'application/json','Accept-Charset':'UTF-8'}
    r=requests.post(url,data=j_data,headers=headers)
    r1=list(r.text)
    stat=""
    s=diabetes_risk_prediction(Glucose,BP,ST,Insulin,BMI,Age)
    if r1[2]==0:
        stat="patient is not affected with Diabetes" 
    else:
        stat="patient affected with Diabetes"
    return render_template("result.html",result=stat,risk=s)
def diabetes_risk_prediction(glucose, bp, skinthickness, insulin, bmi, age):
    indicator_list = [int(glucose), int(bp), int(skinthickness), int(insulin), int(bmi), int(age)]
    predictions = model1.predict_proba(np.array(indicator_list).reshape(1, -1))
    risk = predictions[0,1]
    print("-"*len("Health Indicator Analysis"))
    print("Health Indicator Analysis")
    print("-"*len("Health Indicator Analysis"))
    riskdata=""
    if risk < 0.3:
        riskdata="You are probably in good health, keep it up. \n"

        #print("-"*len("You are probably in good health, keep it up"))
            
    elif risk > 0.9:
        riskdata="Go to a hospital right away. Odds are high you have diabetes."
    elif risk > 0.7:
        riskdata="See a doctor as soon as you can and listen to their recommendations. You might be on the way to developing diabetes if you don't change your lifestyle."
    else:
        riskdata="You should be alright for the most part, but take care not to let your health slip. \n"
    return riskdata +"\n" + "Your Diabetes Risk Index is {:.2f}/100.".format(risk*100)

if __name__=='__main__':
    modelfile='final_diabetes.pickle'
    modelfile1='final_risk.pickle'
    model=p.load(open(modelfile,'rb'))
    model1=p.load(open(modelfile1,'rb'))
    app.run(debug=True,host='0.0.0.0')
