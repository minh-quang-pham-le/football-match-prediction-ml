import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from src.data.data_preprocessing import *

# 1. Load data
df = pd.read_csv('data/raw/df_2.csv', parse_dates=['date'])

# 2. Chia dữ liệu thành train, val, test
X_train, y_train, X_val, y_val, X_test, y_test = split_train_test(df)

# 3. Tính định các cột chứa giá trị null và tính toán các giá trị này
odds_cols       = [c for c in X_train.columns if c.startswith(('B365','BW','LB','WH','VC'))]
tactic_cols     = [
    *[f'home_{p}' for p in ['buildUpPlaySpeed','buildUpPlayPassing','chanceCreationPassing','chanceCreationShooting','defencePressure','defenceAggression','defenceTeamWidth']],
    *[f'away_{p}' for p in ['buildUpPlaySpeed','buildUpPlayPassing','chanceCreationPassing','chanceCreationShooting','defencePressure','defenceAggression','defenceTeamWidth']]
]
player_attr_cols = [c for c in X_train.columns if c.startswith(('home_avg_','away_avg_','home_overall_rating','away_overall_rating'))]

odds_imp    = ThreeTierGroupImputer(odds_cols,    ['league_id','season'],     '_odds_missing')
tactic_imp  = ThreeTierGroupImputer(tactic_cols,  ['league_id','prev_season'], '_tactics_missing')
attr_imp    = GlobalMedianImputer(player_attr_cols,  '_player_attr_missing')
diff_transf = FunctionTransformer(recalc_diff, validate=False)

# Fit imputers trên tập train
for imp in (odds_imp, tactic_imp, attr_imp):
    imp.fit(X_train)
    
# Transform riêng trên tập train để lưu ra file csv để EDA
# Transform để điền thiếu
X_train_imputed = X_train.copy()
X_train_imputed = odds_imp.transform(X_train_imputed)
X_train_imputed = tactic_imp.transform(X_train_imputed)
X_train_imputed = attr_imp.transform(X_train_imputed)

# Tính lại diff
X_train_imputed = recalc_diff(X_train_imputed)

# Gắn lại nhãn
X_train_imputed['outcome'] = y_train

# Lưu ra file CSV để làm EDA
import os
if not os.path.exists("data/feature"):
    os.makedirs("data/feature", exist_ok=True)
X_train_imputed.to_csv("data/feature/X_train_imputed_for_eda.csv", index=False)

# Các cột cho việc encoding và scaling
id_cols   = ['league_id','home_team_api_id','away_team_api_id']
flag_cols = [c for c in X_train.columns if c.endswith('_missing')]
cat_cols  = ['match_phase','stage']
num_cols  = [c for c in X_train.columns if c not in id_cols+flag_cols+cat_cols+['outcome','season','prev_season']]

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

# 5. Full hyperparameter grid
BIG_GRID = [
    {
     'model': [LGBMClassifier(objective='multiclass', n_jobs=-1, random_state=42)],
     'prep' : [pre_tree],
     'model__num_leaves':       [31,63,127,255,511],
     'model__max_depth':        [-1,5,7,9,12],
     'model__learning_rate':    [0.3,0.1,0.05,0.03,0.01],
     'model__n_estimators':     [400,800,1200,1600,2000],
     'model__subsample':        [1.0,0.8,0.6,0.4,0.3],
     'model__colsample_bytree': [1.0,0.8,0.6,0.4,0.3]
    },
    {
     'model': [XGBClassifier(objective='multi:softprob', n_jobs=-1, eval_metric='mlogloss', random_state=42)],
     'prep' : [pre_tree],
     'model__max_depth':        [3,5,7,9,12],
     'model__learning_rate':    [0.3,0.1,0.05,0.03,0.01],
     'model__n_estimators':     [300,500,700,1000,1300],
     'model__subsample':        [1.0,0.9,0.8,0.7,0.6],
     'model__colsample_bytree': [1.0,0.9,0.8,0.7,0.6]
    },
    {
     'model': [RandomForestClassifier(n_jobs=-1, random_state=42, class_weight='balanced')],
     'prep' : [pre_tree],
     'model__n_estimators':     [200,400,800,1200,1600],
     'model__max_depth':        [None,10,15,20,25],
     'model__min_samples_split':[2,4,6,10,20],
     'model__min_samples_leaf': [1,2,4,8,12]
    },
    {
     'model': [GradientBoostingClassifier(random_state=42)],
     'prep' : [pre_tree],
     'model__learning_rate':    [0.1,0.05,0.03,0.01,0.005],
     'model__n_estimators':     [200,400,600,800,1000],
     'model__max_depth':        [3,4,5,6,7]
    },
    {
     'model': [LogisticRegression(max_iter=3000, multi_class='multinomial', n_jobs=-1)],
     'prep' : [pre_lin],
     'model__C':                [0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000],
     'model__penalty':          ['l2','l1','elasticnet'],
     'model__solver':           ['saga']
    },
    {
     'model': [SVC(probability=True)],
     'prep' : [pre_lin],
     'model__C':      [0.1,0.3,1,3,10,30,100,300,1000,3000,10000],
     'model__kernel': ['rbf','poly'],
     'model__gamma':  ['scale','auto',1e-2,1e-3,1e-4],
     'model__degree': [2,3,4]
    }
]

# 6. GridSearch với imputation + prep + model
results = {}
tscv = TimeSeriesSplit(n_splits=5)

for spec in BIG_GRID:
    mdl = spec.pop('model')[0]
    prep = spec.pop('prep')[0]
    pipe = Pipeline([
        ('imp_odds',   odds_imp),
        ('imp_tactic', tactic_imp),
        ('imp_attr',   attr_imp),
        ('diff',       diff_transf),
        ('prep',       prep),
        ('model',      mdl)
    ])
    gcv = GridSearchCV(pipe, spec, cv=tscv, scoring='accuracy', n_jobs=-1, verbose=2)
    name = mdl.__class__.__name__
    print(f"\n>>> Training {name}")
    gcv.fit(X_train, y_train)
    print(f" Best params: {gcv.best_params_}")
    yv = gcv.predict(X_val)
    print(f"\nValidation report ({name}):")
    print(classification_report(y_val, yv, digits=4))
    results[name] = (gcv.best_estimator_, classification_report(y_val, yv, output_dict=True))

# 7. Đánh giá trên tập test
best = max(results, key=lambda k: results[k][1]['accuracy'])
best_pipe = results[best][0]
print(f"\n>>> Testing best model: {best}")
yt = best_pipe.predict(X_test)
print(classification_report(y_test, yt, digits=4))