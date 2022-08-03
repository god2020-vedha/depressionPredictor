# Importing necessary libraries
import numpy as np
from flask import Flask, request, make_response
from multiprocessing import Process
import json
import pickle
from flask_cors import cross_origin

# Declaring the flask app
app = Flask(__name__)

#Loading the model from pickle file
model = pickle.load(open('lr_trained_model.sav','rb'))
@app.route('/')
@app.route('/home')
def home():
    return "WELCOME DIANA"
# geting and sending response to dialogflow
@app.route('/webhook', methods=['POST'])
#@cross_origin()
def webhook():
    req = request.get_json(silent=True, force=True)
    res = processRequest(req)
    res = json.dumps(res, indent=4)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    #req = request.get_json(force=True)
    print(r)
    return r
# processing the request from dialogflow
def processRequest(req):
    result = req.get("queryResult")
    #Fetching the data points
    parameters = result.get("parameters")
    gender=parameters.get("gender")
    age = parameters.get("age")
    edu=parameters.get("edu")
    inc=parameters.get("inc")
    trsleep=parameters.get("trsleep")
    sleephrs=parameters.get("sleephrs")
    sed=parameters.get("sed")
    work=parameters.get("work")
    lim=parameters.get("lim")
    mem=parameters.get("mem")
    pres=parameters.get("pres")
    int_features = [gender,age,edu,inc,trsleep,sleephrs, sed, work,lim,mem,pres]
    #Dumping the data into an array
    final_features = [np.array(int_features)]
    
    #Getting the intent which has fullfilment enabled
    intent = result.get("intent").get('displayName')
    
    #Fitting out model with the data points
    if (intent=='DepData'):
        prediction = model.predict(final_features)
        print(prediction)
        #output=prediction
        output = round(prediction[0], 2)
       	
        if(output== 0):
            state = 'Not Depressed'
    
        else:
            
            state = 'Depressed'
    
                   
        #Returning back the fullfilment text back to DialogFlow
        fulfillmentText= "Your Depression Status is.  {} !".format(state)
        #log.write_log(sessionID, "Bot Says: "+fulfillmentText)
        return { "fulfillmentText": fulfillmentText}

if __name__ == '__main__':
    app.secret_key = 'ItIsASecret'
    app.debug = True
    app.run(host="0.0.0.0",port=5001)