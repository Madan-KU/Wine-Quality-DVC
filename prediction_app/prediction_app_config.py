import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, 'models', 'best_model_rf_bayes.pkl')
SCALER_PATH = os.path.join(CURRENT_DIR, 'scaler', 'scaler.pkl')
PARAMS_PATH = "params.yaml"
STATIC_PATH = os.path.join(CURRENT_DIR, "static")
TEMPLATE_PATH = os.path.join(CURRENT_DIR, "templates")
