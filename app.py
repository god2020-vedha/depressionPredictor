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
model = pickle.load(open('rf1.pkl','rb'))
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
    Petal_length=parameters.get("number")
    Petal_width = parameters.get("number1")
    Sepal_length=parameters.get("number2")
    Sepal_width=parameters.get("number3")
    int_features = [Petal_length,Petal_width,Sepal_length,Sepal_width]
    #Dumping the data into an array
    final_features = [np.array(int_features)]
    
    #Getting the intent which has fullfilment enabled
    intent = result.get("intent").get('displayName')
    
    #Fitting out model with the data points
    if (intent=='IrisData'):
        prediction = model.predict(final_features)
        print(prediction)
        output=prediction
        #output = round(prediction[0], 2)
       	
        if(output=='Iris-setosa'):
            flowr = 'Setosa'
    
        if(output=="Iris-versicolor"):
            flowr = 'Versicolour'
        
        if(output=="Iris-virginica"):
            flowr = 'Virginica'
            
        #Returning back the fullfilment text back to DialogFlow
        fulfillmentText= "The Iris type seems to be..  {} !".format(flowr)
        #log.write_log(sessionID, "Bot Says: "+fulfillmentText)
        return { "fulfillmentText": fulfillmentText}


if __name__ == '__main__':
    app.secret_key = 'ItIsASecret'
    app.debug = True
    app.run(port=5003)