import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def main():
    # 1) Load dữ liệu
    X_train = pd.read_csv("data/feature/X_train.csv")
    y_train = pd.read_csv("data/feature/y_train.csv").squeeze()
    X_val   = pd.read_csv("data/feature/X_val.csv")
    y_val   = pd.read_csv("data/feature/y_val.csv").squeeze()
    X_test  = pd.read_csv("data/feature/X_test.csv")
    y_test  = pd.read_csv("data/feature/y_test.csv").squeeze()
    
    # 2) Column groups
    id_cols   = ['league_id','home_team_api_id','away_team_api_id']
    flag_cols = [c for c in X_train.columns if c.endswith('_missing')]
    cat_cols  = ['match_phase','stage']
    num_cols  = [c for c in X_train.columns 
                 if c not in (*id_cols, *flag_cols, *cat_cols)]
    
    # 3) Preprocessors
    pre_tree = ColumnTransformer([
        ('id',    OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), id_cols),
        ('cat',   OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
        ('pas',   'passthrough', flag_cols + num_cols)
    ])
    pre_lin  = ColumnTransformer([
        ('id',    OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), id_cols),
        ('cat',   OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
        ('scale', StandardScaler(), num_cols),
        ('pas',   'passthrough', flag_cols)
    ])
    
    # 4) Hyperparameter grid
    BIG_GRID = [
        # LightGBM
        { 'model': [LGBMClassifier(objective='multiclass', n_jobs=-1, random_state=42)],
          'prep' : [pre_tree],
          'model__num_leaves':       [31,63,127,255,511],
          'model__max_depth':        [-1,5,7,9,12],
          'model__learning_rate':    [0.3,0.1,0.05,0.03,0.01],
          'model__n_estimators':     [400,800,1200,1600,2000],
          'model__subsample':        [1.0,0.8,0.6,0.4,0.3],
          'model__colsample_bytree': [1.0,0.8,0.6,0.4,0.3]
        },
        # XGBoost
        { 'model': [XGBClassifier(objective='multi:softprob', n_jobs=-1,
                                  eval_metric='mlogloss', random_state=42)],
          'prep' : [pre_tree],
          'model__max_depth':        [3,5,7,9,12],
          'model__learning_rate':    [0.3,0.1,0.05,0.03,0.01],
          'model__n_estimators':     [300,500,700,1000,1300],
          'model__subsample':        [1.0,0.9,0.8,0.7,0.6],
          'model__colsample_bytree': [1.0,0.9,0.8,0.7,0.6]
        },
        # Random Forest
        { 'model': [RandomForestClassifier(n_jobs=-1, random_state=42,
                                           class_weight='balanced')],
          'prep' : [pre_tree],
          'model__n_estimators':     [200,400,800,1200,1600],
          'model__max_depth':        [None,10,15,20,25],
          'model__min_samples_split':[2,4,6,10,20],
          'model__min_samples_leaf': [1,2,4,8,12]
        },
        # Gradient Boosting
        { 'model': [GradientBoostingClassifier(random_state=42)],
          'prep' : [pre_tree],
          'model__learning_rate':    [0.1,0.05,0.03,0.01,0.005],
          'model__n_estimators':     [200,400,600,800,1000],
          'model__max_depth':        [3,4,5,6,7]
        },
        # Logistic Regression
        { 'model': [LogisticRegression(max_iter=3000, multi_class='multinomial',
                                       n_jobs=-1)],
          'prep' : [pre_lin],
          'model__C':                [0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000],
          'model__penalty':          ['l2','l1','elasticnet'],
          'model__solver':           ['saga']
        },
        # SVM
        { 'model': [SVC(probability=True)],
          'prep' : [pre_lin],
          'model__C':      [0.1,0.3,1,3,10,30,100,300,1000,3000,10000],
          'model__kernel': ['rbf','poly'],
          'model__gamma':  ['scale','auto',1e-2,1e-3,1e-4],
          'model__degree': [2,3,4]
        }
    ]
    
    # 5) GridSearch + TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    results = {}
    for spec in BIG_GRID:
        model = spec.pop('model')[0]
        prep  = spec.pop('prep')[0]
        pipe  = Pipeline([('prep', prep), ('model', model)])
        
        gcv = GridSearchCV(pipe, spec,
                           cv=tscv,
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=2)
        name = model.__class__.__name__
        print(f"\n>>> Training {name}")
        gcv.fit(X_train, y_train)
        
        yv = gcv.predict(X_val)
        print(f"\nValidation report ({name}):")
        print(classification_report(y_val, yv, digits=4))
        
        results[name] = (gcv.best_estimator_,
                         classification_report(y_val, yv, output_dict=True))
    
    # 6) Final test
    best = max(results, key=lambda k: results[k][1]['accuracy'])
    best_pipe = results[best][0]
    print(f"\n>>> Testing best model: {best}")
    yt = best_pipe.predict(X_test)
    print(classification_report(y_test, yt, digits=4))

if __name__ == "__main__":
    main()