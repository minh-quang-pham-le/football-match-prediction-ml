import warnings, os, joblib, time
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier, early_stopping
from xgboost import XGBClassifier, callback
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint, uniform
import json
warnings.filterwarnings('ignore')

def load_data():
    X_train = pd.read_csv("data/feature/X_train.csv")
    y_train = pd.read_csv("data/feature/y_train.csv").squeeze()
    X_val   = pd.read_csv("data/feature/X_val.csv")
    y_val   = pd.read_csv("data/feature/y_val.csv").squeeze()
    X_test  = pd.read_csv("data/feature/X_test.csv")
    y_test  = pd.read_csv("data/feature/y_test.csv").squeeze()
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_preprocessors(X):
    id_cols   = ['league_id','home_team_api_id','away_team_api_id']
    flag_cols = [c for c in X.columns if c.endswith('_missing')]
    cat_cols  = ['match_phase','stage']
    num_cols  = [c for c in X.columns if c not in (*id_cols, *flag_cols, *cat_cols)]

    pre_tree = ColumnTransformer([
        ('id',  OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), id_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
        ('pas', 'passthrough', flag_cols + num_cols)
    ])
    pre_lin = ColumnTransformer([
        ('id',  OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), id_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
        ('sc',  StandardScaler(), num_cols),
        ('pas', 'passthrough', flag_cols)
    ])
    return pre_tree, pre_lin
  
def prefix_fit_params(step_name: str, params: dict):
    return {f"{step_name}__{k}": v for k, v in params.items()}

# Helper cân bằng lớp
def make_class_weight(y):
    freq = y.value_counts(normalize=True).sort_index()
    inv  = 1. / freq
    # Mean-scaling
    w = (inv / inv.mean()).to_dict()
    # dict, sample_weight vec          
    return w, y.map(w).values 

def main():
    # 1) Load dữ liệu
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # 2) Khởi tạo các preprocessors
    pre_tree, pre_lin = build_preprocessors(X_train)
    
    cls_weight_dict, sample_weight_train = make_class_weight(y_train)
    
    # 3) Xác định các model và hyperparameters
    models = {
        "LightGBM": (
            LGBMClassifier(objective='multiclass', metric='multi_logloss', 
                           class_weight = cls_weight_dict, device='gpu', random_state=42),
            { "model__num_leaves":       randint(31, 512),
              "model__max_depth":        [-1, 7, 11, 15, 19],
              "model__learning_rate":    uniform(0.01, 0.29),
              "model__n_estimators":     randint(200, 801),
              "model__min_child_samples":randint(10, 200) },
            pre_tree,
            {"eval_set":[(X_val,y_val)], "callbacks":[early_stopping(50, first_metric_only=True, verbose=False)]}
        ),
        "XGBoost": (
            XGBClassifier(objective='multi:softprob',
                          tree_method='gpu_hist', eval_metric='mlogloss',
                          random_state=42),
            { "model__max_depth":        randint(3, 12),
              "model__learning_rate":    uniform(0.01, 0.29),
              "model__n_estimators":     randint(200, 800),
              "model__subsample":        uniform(0.6, 0.4),
              "model__colsample_bytree": uniform(0.6, 0.4) },
            pre_tree,
            {"eval_set":[(X_val,y_val)], "callbacks":[callback.EarlyStopping(rounds=50, metric='mlogloss', save_best=True, verbose=False)]}
        ),
        "RandomForest": (
            RandomForestClassifier(class_weight=cls_weight_dict,
                                   n_jobs=-1, random_state=42),
            { "model__n_estimators":     randint(200,800),
              "model__max_depth":        [None, 10, 20, 30],
              "model__min_samples_split":randint(2,20),
              "model__min_samples_leaf": randint(1,10) },
            pre_tree,
            {}
        ),
        "GradientBoost": (
            GradientBoostingClassifier(random_state=42),
            { "model__learning_rate":    uniform(0.01,0.29),
              "model__n_estimators":     randint(100,600),
              "model__max_depth":        randint(2,7),
              "model__subsample":        uniform(0.5,0.5) },
            pre_tree,
            {}
        ),
        "LogisticRegression": (
            LogisticRegression(max_iter=4000, multi_class='multinomial',
                               n_jobs=-1),
            { "model__C":        uniform(0.01, 100),
              "model__penalty":  ['l2','none'],
              "model__solver":   ['lbfgs','saga'] },
            pre_lin,
            {}
        )
    }
    
    # 4) Train, validate các model, và lưu lại model tốt nhất trên tập val
    tscv   = TimeSeriesSplit(n_splits=3)
    best_pipe, best_score, best_name = None, -1, ""
    
    os.makedirs("models", exist_ok=True)
    
    for name,(est,dist,prep,fit_kw) in models.items():
        n_iter = 30 if name in {"RandomForest", "GradientBoost"} else 50
        print(f"\n▶▶ {name}: Randomized search ({n_iter} configs, 3-fold)")
        
        pipe = Pipeline([('prep', prep), ('model', est)])
        fit_params = prefix_fit_params('model', fit_kw)
        search = RandomizedSearchCV(
            pipe, dist, n_iter=n_iter, cv=tscv,
            scoring='accuracy', n_jobs=-1, random_state=42, verbose=1, refit=True
        )
        
        tic = time.time()
        search.fit(X_train, y_train, **fit_params)
        toc = time.time()
        print(f"↳ Done in {(toc-tic)/60:.1f} min — best Acc={search.best_score_:.4f}")

        # Lưu hyperparameters tốt nhất
        with open(f"models/{name}_best.json", "w") as fp:
          json.dump(search.best_params_, fp, indent=2, default=str)

        # In top-5 cấu hình tốt nhất
        top5 = (pd.DataFrame(search.cv_results_)
              .sort_values("rank_test_score")
              .head(5)[["mean_test_score", "params"]])
        print(top5.to_string(index=False))
        
        # Đánh giá trên tập val (season 2014/2015)
        y_pred = search.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(classification_report(y_val, y_pred, digits=4))

        # Cập nhật best
        if acc > best_score:
            best_score, best_pipe, best_name = acc, search.best_estimator_, name

        # Lưu checkpoint (chứa cả pipeline + preprocessors)
        joblib.dump(search.best_estimator_, f"models/{name}_best.pkl")
    
    # 5) Evaluate trên tập test 2015/2016
    if best_pipe is None:
      raise RuntimeError("Không có mô hình nào huấn luyện thành công!")
  
    print(f"\nBest model = {best_name}  (val Acc={best_score:.4f})")
    y_test_pred = best_pipe.predict(X_test)
    print(classification_report(y_test, y_test_pred, digits=4))

if __name__ == "__main__":
  main()