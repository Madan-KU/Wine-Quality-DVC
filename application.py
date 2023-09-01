import os
import redis
import pickle
import joblib
import numpy as np
import pandas as pd

import prediction_app.prediction as pred

from flask import Flask, request, render_template, send_from_directory, redirect, url_for,jsonify

static_folder_path =  os.path.join("prediction_app","static")
template_folder_path= os.path.join("prediction_app","templates")
REDIS_HOST = 'host.docker.internal' #if run within docker
REDIS_HOST ='redis' #if run with docker-compose.yaml

# REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0


# Flask application setup
application = Flask(__name__, static_folder=static_folder_path , template_folder=template_folder_path)

# Connect to Redis AOF
r = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

@application.route('/',methods= ["GET","POST"])
def home():
    print('Accessing the home route...')

    if request.method =="POST":
        
        try:
            if request.form:
                data=dict(request.form).values()
                data=[list(map(float,data))]
                redis_key=str(data)
                # prediction= pred.predict(data) # Remove if used with redis

                #Check if prediction for input data exists in Redis

                if(prediction:=r.get(redis_key)):
                    prediction = prediction + " (from Redis)"
                else:
                    prediction= pred.predict(data)
                    # Store the prediction in Redis
                    r.set(redis_key,prediction) 
           
                return render_template("Predict.html",prediction=prediction)
            
            elif request.json:
                data = np.array(list(request.json.values())) 
                redis_key = str(list(data))
                # prediction = pred.predict([data]).tolist() # Remove if used with redis

                if (prediction := r.get(redis_key)):
                    prediction = [prediction + " (from Redis)"]
                else:
                    prediction = pred.predict([data]).tolist()
                    r.set(redis_key, str(prediction[0]))

                return jsonify({"prediction": prediction})

        
        except FileNotFoundError as e:
            error = f"File not found: {e.filename}"
            return render_template("404.html", error=error)
        
        except Exception as e:
            error=f"An error occurred: {str(e)}"
            return render_template("404.html",error=error)
        
    else:
        return render_template("Predict.html")


if __name__ == '__main__':
    application.run(debug=True,host="0.0.0.0",port=5000)
