import os
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.base import BaseEstimator, TransformerMixin

outcome_to_int = {'Loss': 0, 'Draw': 1, 'Win': 2}   # Win-Loss-Draw -> 2-0-1

def split_train_test(df: pd.DataFrame) -> Tuple[
    pd.DataFrame, pd.Series,  # X_train, y_train
    pd.DataFrame, pd.Series,  # X_val,   y_val
    pd.DataFrame, pd.Series   # X_test,  y_test
]:
    train_seasons = ['2008/2009','2009/2010','2010/2011','2011/2012','2012/2013','2013/2014']
    val_seasons   = ['2014/2015']
    test_seasons  = ['2015/2016']

    # Chia dữ liệu thành train, val, test và loại bỏ cột 'season'   
    df_train = df[df['season'].isin(train_seasons)].reset_index(drop=True)
    df_val   = df[df['season'].isin(val_seasons)].reset_index(drop=True)
    df_test  = df[df['season'].isin(test_seasons)].reset_index(drop=True)

    # Tách X và y
    X_train, y_train = df_train.drop(columns=['outcome']), df_train['outcome'].map(outcome_to_int).astype('int8')
    X_val,   y_val   = df_val.drop(columns=['outcome']),   df_val['outcome'].map(outcome_to_int).astype('int8')
    X_test,  y_test  = df_test.drop(columns=['outcome']),  df_test['outcome'].map(outcome_to_int).astype('int8')

    return X_train, y_train, X_val, y_val, X_test, y_test

# Tính toán các giá trị null của các features
# Imputer transformers
class ThreeTierGroupImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, group_by, flag_suffix):
        self.columns = columns
        self.group_by = group_by
        self.flag_suffix = flag_suffix

    def fit(self, X, y=None):
        self.group_medians_ = {
            col: X.groupby(self.group_by)[col].median()
            for col in self.columns
        }
        self.global_medians_ = {
            col: X[col].median() for col in self.columns
        }
        return self

    def transform(self, X):
        X = X.copy()
        # Tạo key là tuple các giá trị của group_by
        grp_key = X[self.group_by].apply(lambda r: tuple(r), axis=1)
        for col in self.columns:
            # flag missing
            X[f"{col}{self.flag_suffix}"] = X[col].isna().astype(int)
            # lấy Series median với MultiIndex
            grp_med = self.group_medians_[col]
            # điền NaN theo group median rồi global median
            X[col] = (
                X[col]
                 .fillna(grp_key.map(grp_med))
                 .fillna(self.global_medians_[col])
            )
        return X

class GlobalMedianImputer(BaseEstimator, TransformerMixin):
    """
    Impute by global median and add missing-flag.
    """
    def __init__(self, columns, flag_suffix):
        self.columns = columns
        self.flag_suffix = flag_suffix

    def fit(self, X, y=None):
        self.global_medians_ = {col: X[col].median() for col in self.columns}
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[f"{col}{self.flag_suffix}"] = X[col].isna().astype(int)
            X[col] = X[col].fillna(self.global_medians_[col])
        return X

# diff recalculation
def recalc_diff(df):
    diff_map = {
    'diff_speed': ('home_buildUpPlaySpeed', 'away_buildUpPlaySpeed'),
    'diff_shooting': ('home_chanceCreationShooting', 'away_chanceCreationShooting'),
    'diff_pressure': ('home_defencePressure', 'away_defencePressure'),
}
    df = df.copy()
    for dcol, (hcol, acol) in diff_map.items():
        df[dcol] = df[hcol] - df[acol]
        df.fillna({dcol: 0}, inplace=True)
    return df

def main():
    # 1) Load data
    df = pd.read_csv('data/processed/df_2.csv')
    
    # 2) Split theo mùa
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_test(df)
    
    # 3) Khởi tạo imputers
    odds_cols    = [c for c in X_train if c.startswith(('B365','BW','LB','WH','VC','IW'))]
    tactic_cols  = [
        *[f"home_{p}" for p in ['buildUpPlaySpeed','buildUpPlayPassing',
                                 'chanceCreationPassing','chanceCreationShooting',
                                 'defencePressure','defenceAggression','defenceTeamWidth']],
        *[f"away_{p}" for p in ['buildUpPlaySpeed','buildUpPlayPassing',
                                 'chanceCreationPassing','chanceCreationShooting',
                                 'defencePressure','defenceAggression','defenceTeamWidth']]
    ]
    player_cols  = [c for c in X_train if c.startswith(('home_avg_','away_avg_'))]
    
    odds_imp   = ThreeTierGroupImputer(odds_cols,   ['league_id','season'],     '_odds_missing')
    tact_imp   = ThreeTierGroupImputer(tactic_cols, ['league_id','prev_season'], '_tactics_missing')
    attr_imp   = GlobalMedianImputer(player_cols,   '_player_attr_missing')
    
    # 4) Fit & transform train, val, test
    for imp in (odds_imp, tact_imp, attr_imp):
        imp.fit(X_train)
        for X in (X_train, X_val, X_test):
            X[:] = imp.transform(X)
    
    # 5) Recalc diff trên cả 3
    X_train[:] = recalc_diff(X_train)
    X_val[:]   = recalc_diff(X_val)
    X_test[:]  = recalc_diff(X_test)
    
    # 6) Drop cột 'season' và 'prev_season' trên cả 3 tập train, val, và test
    X_train.drop(columns=['season','prev_season'], inplace=True)
    X_val.drop(columns=['season','prev_season'], inplace=True)
    X_test.drop(columns=['season','prev_season'], inplace=True)
    
    # 7) Tạo folder nếu chưa có
    os.makedirs(os.path.join("data","feature"), exist_ok=True)
    
    # 8) Lưu CSV
    X_train.to_csv("data/feature/X_train.csv", index=False)
    y_train.to_csv("data/feature/y_train.csv", index=False, header=True)
    X_val  .to_csv("data/feature/X_val.csv",   index=False)
    y_val  .to_csv("data/feature/y_val.csv",   index=False, header=True)
    X_test .to_csv("data/feature/X_test.csv",  index=False)
    y_test .to_csv("data/feature/y_test.csv",  index=False, header=True)
    
    print("[DONE] Preprocessing data đã xong, dữ liệu được lưu vào data/feature/")

if __name__ == "__main__":
    main()