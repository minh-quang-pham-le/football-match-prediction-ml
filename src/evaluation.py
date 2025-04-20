import joblib
from sklearn.metrics import classification_report

def evaluate_model(model_path, X_test, y_test):
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))