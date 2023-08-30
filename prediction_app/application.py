import os
import redis
import pickle
import joblib
import numpy as np
import pandas as pd

import prediction_app_config as config
import prediction as pred

from flask import Flask, request, render_template, send_from_directory, redirect, url_for,jsonify

# Flask application setup
application = Flask(__name__, static_folder=config.path.static_path, template_folder=config.path.template_path)

# Connect to Redis AOF
# r = redis.StrictRedis(host=config.redis.REDIS_HOST, port=config.redis.REDIS_PORT, db=config.redis.REDIS_DB, decode_responses=True)

@application.route('/',methods= ["GET","POST"])
def home():
    print('Accessing the home route...')

    if request.method =="POST":
        
        try:
            if request.form:
                data=dict(request.form).values()
                data=[list(map(float,data))]
                # redis_key=str(data)
                prediction= pred.predict(data) # Remove if used with redis

                # #Check if prediction for input data exists in Redis

                # if(prediction:=r.get(redis_key)):
                #     prediction = prediction + " (from Redis)"
                # else:
                #     prediction= pred.predict(data)
                #     # Store the prediction in Redis
                #     r.set(redis_key,prediction) 
           
                return render_template("Predict.html",prediction=prediction)
            
            elif request.json:
                data = np.array(list(request.json.values())) 
                # redis_key = str(list(data))
                prediction = pred.predict([data]).tolist() # Remove if used with redis

                # if (prediction := r.get(redis_key)):
                #     prediction = [prediction + " (from Redis)"]
                # else:
                #     prediction = pred.predict([data]).tolist()
                #     r.set(redis_key, str(prediction[0]))

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
