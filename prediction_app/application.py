import os
import pickle
import pandas as pd
from flask import Flask, request, render_template, send_from_directory, redirect, url_for,jsonify
import joblib
import numpy as np
import prediction_app_config as app_config
import prediction as pred


# Flask application setup
application = Flask(__name__, static_folder=app_config.STATIC_PATH, template_folder=app_config.TEMPLATE_PATH)


@application.route('/',methods= ["GET","POST"])
def home():
    print('Accessing the home route...')

    if request.method =="POST":
        
        try:
            if request.form:
                data=dict(request.form).values()
                data=[list(map(float,data))]
                prediction= pred.predict(data)
                return render_template("Predict.html",prediction=prediction)
            
            elif request.json:
                prediction=api_response(request)
                return jsonify(prediction)

        
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
