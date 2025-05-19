import os
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.base import BaseEstimator, TransformerMixin

def split_train_test(df: pd.DataFrame) -> Tuple[
    pd.DataFrame, pd.Series,  # X_train, y_train
    pd.DataFrame, pd.Series,  # X_val,   y_val
    pd.DataFrame, pd.Series   # X_test,  y_test
]:
    train_seasons = ['2008/2009','2009/2010','2010/2011','2011/2012','2012/2013','2013/2014']
    val_seasons   = ['2014/2015']
    test_seasons  = ['2015/2016']

    # Chia dữ liệu thành train, val, test và loại bỏ cột 'season'   
    df_train = df[df['season'].isin(train_seasons)].reset_index(drop=True).drop(columns=['season'])
    df_val   = df[df['season'].isin(val_seasons)].reset_index(drop=True).drop(columns=['season'])
    df_test  = df[df['season'].isin(test_seasons)].reset_index(drop=True).drop(columns=['season'])

    # Tách X và y
    X_train, y_train = df_train.drop(columns=['outcome']), df_train['outcome']
    X_val,   y_val   = df_val.drop(columns=['outcome']),   df_val['outcome']
    X_test,  y_test  = df_test.drop(columns=['outcome']),  df_test['outcome']

    return X_train, y_train, X_val, y_val, X_test, y_test

# Tính toán các giá trị null của các features
# Imputer transformers
class ThreeTierGroupImputer(BaseEstimator, TransformerMixin):
    """
    Impute by:
      1) median within (group_by)
      2) global median fallback
    Also adds a missing-flag column for each feature.
    """
    def __init__(self, columns, group_by, flag_suffix):
        self.columns = columns
        self.group_by = group_by
        self.flag_suffix = flag_suffix

    def fit(self, X, y=None):
        # compute per-group and global medians
        self.group_medians_ = {col: X.groupby(self.group_by)[col].median() for col in self.columns}
        self.global_medians_ = {col: X[col].median() for col in self.columns}
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[f"{col}{self.flag_suffix}"] = X[col].isna().astype(int)
            grp = X[self.group_by].map(self.group_medians_[col])
            X[col] = X[col].fillna(grp).fillna(self.global_medians_[col])
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
        df[dcol].fillna(0, inplace=True)
    return df

def main():
    # 1) Load raw data
    df = pd.read_csv('data/processed/df_2.csv')
    
    # 2) Split theo mùa
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_test(df)
    
    # 3) Khởi tạo imputers
    odds_cols    = [c for c in X_train if c.startswith(('B365','BW','LB','WH','VC'))]
    tactic_cols  = [
        *[f"home_{p}" for p in ['buildUpPlaySpeed','buildUpPlayPassing',
                                 'chanceCreationPassing','chanceCreationShooting',
                                 'defencePressure','defenceAggression','defenceTeamWidth']],
        *[f"away_{p}" for p in ['buildUpPlaySpeed','buildUpPlayPassing',
                                 'chanceCreationPassing','chanceCreationShooting',
                                 'defencePressure','defenceAggression','defenceTeamWidth']]
    ]
    player_cols  = [c for c in X_train if c.startswith(('home_avg_','away_avg_'
                                                        ,'home_overall_rating','away_overall_rating'))]
    
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
    
    # 6) Tạo folder nếu chưa có
    os.makedirs(os.path.join("data","feature"), exist_ok=True)
    
    # 7) Lưu CSV
    X_train.to_csv("data/feature/X_train.csv", index=False)
    y_train.to_csv("data/feature/y_train.csv", index=False, header=True)
    X_val  .to_csv("data/feature/X_val.csv",   index=False)
    y_val  .to_csv("data/feature/y_val.csv",   index=False, header=True)
    X_test .to_csv("data/feature/X_test.csv",  index=False)
    y_test .to_csv("data/feature/y_test.csv",  index=False, header=True)
    
    print("Preprocessing done, features saved to data/feature/")

if __name__ == "__main__":
    main()