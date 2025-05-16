import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    '''Đánh giá mô hình với các chỉ số Accuracy, F1-score, AUC-ROC và Classification Report.'''
    
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred, average='macro'))
    print("AUC-ROC:", roc_auc_score(pd.get_dummies(y_test), model.predict_proba(X_test), multi_class='ovr'))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))