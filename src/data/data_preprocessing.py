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
    
    df_train = df[df['season'].isin(train_seasons)].reset_index(drop=True)
    df_val   = df[df['season'].isin(val_seasons)].reset_index(drop=True)
    df_test  = df[df['season'].isin(test_seasons)].reset_index(drop=True)

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