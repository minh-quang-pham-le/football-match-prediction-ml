import xgboost as xgb
import joblib

def train_model(X_train, y_train):
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    # joblib.dump(model, 'models/')
    #return model
    pass
