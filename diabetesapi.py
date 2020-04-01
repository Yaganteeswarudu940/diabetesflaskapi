from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import sklearn
import json
import pickle as p
app=Flask(__name__)
@app.route("/diabetesprediction", methods=['POST'])
def predictdiabetes():
    data=request.get_json()
    prediction=np.array2string(model.predict(data))
    return jsonify(prediction)
if __name__=='__main__':
    modelfile='final_diabetes.pickle'
    model=p.load(open(modelfile,'rb'))
    app.run(debug=True,host='0.0.0.0')







        
