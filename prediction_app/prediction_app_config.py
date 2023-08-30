import os

class path:
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'models', 'best_model_rf_bayes.pkl')
    scaler_path = os.path.join(base_path, 'scaler', 'scaler.pkl')
    params_path = "params.yaml"
    static_path = os.path.join(base_path, "static")
    template_path = os.path.join(base_path, "templates")

class redis:
    REDIS_HOST = 'localhost'
    REDIS_PORT = 6379
    REDIS_DB = 0
