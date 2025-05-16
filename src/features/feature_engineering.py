import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import OneHotEncoder

def calculate_league_avg_goals(matches, split_year) -> dict:
    '''Tính trung bình bàn thắng trên sân nhà theo mùa giải (chỉ tính trên tập train để tránh data leakage).'''
    
    train_matches = matches[matches['date'].dt.year <= split_year]
    if train_matches.empty:
        print("Warning: No matches found before split_year. Returning default average of 0.")
        return {'default': 0}
    league_avg = train_matches.groupby('season')['home_team_goal'].mean().to_dict()
    return league_avg

def calculate_recent_goals(matches, league_avg_goals) -> pd.DataFrame:
    '''Tính trung bình số bàn thắng dựa trên lịch sử đấu của đội'''
    
    matches = matches.sort_values('date')
    
    # Hàm tính trung bình có trọng số trong cửa sổ trượt
    def weighted_rolling_mean(x, weights):
        if len(x) >= 5:
            return (x * weights).sum() / weights.sum()
        elif len(x) > 0:
            adjusted_weights = weights[-len(x):]
            return (x * adjusted_weights).sum() / adjusted_weights.sum()
        else:
            return league_avg_goals.get(x.name[1], 0)  # season
    
    weights = np.array([0.4, 0.3, 0.2, 0.1, 0.05])
    
    # Tính home_recent_goals
    matches['home_recent_goals'] = matches.groupby('home_team_api_id')['home_team_goal'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).apply(
            lambda y: weighted_rolling_mean(y, weights), raw=False
        )
    )
    matches['home_recent_goals'] = matches.apply(
        lambda row: row['home_recent_goals'] if pd.notnull(row['home_recent_goals']) 
        else league_avg_goals.get(row['season'], 0), axis=1
    )
    
    # Tính away_recent_goals
    matches['away_recent_goals'] = matches.groupby('away_team_api_id')['away_team_goal'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).apply(
            lambda y: weighted_rolling_mean(y, weights), raw=False
        )
    )
    matches['away_recent_goals'] = matches.apply(
        lambda row: row['away_recent_goals'] if pd.notnull(row['away_recent_goals']) 
        else league_avg_goals.get(row['season'], 0), axis=1
    )
    
    return matches

def get_nearest_attributes(match_date, team_api_id, attributes_df) -> pd.Series:
    '''Lấy tất cả bản ghi team_attributes trước ngày trận đấu, tính trung bình cho cột số, lấy gần nhất cho cột nominal.'''
    
    # Lọc các bản ghi trước ngày trận đấu
    team_attrs = attributes_df[(attributes_df['team_api_id'] == team_api_id) & (attributes_df['date'] < match_date)
    ].sort_values('date', ascending=False)
    
    # Nếu không có bản ghi nào, trả về giá trị rỗng
    if team_attrs.empty:
        return pd.Series(index=attributes_df.columns.drop(['team_api_id', 'date']), dtype=float)
    
    # Phân loại cột số và nominal
    numeric_cols = team_attrs.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = team_attrs.select_dtypes(include=['object']).columns
    
    # Tính trung bình cho cột số
    numeric_avg = team_attrs[numeric_cols].mean()
    
    # Lấy giá trị gần nhất cho cột nominal
    nearest_record = team_attrs.iloc[0]  # Bản ghi gần nhất
    categorical_nearest = nearest_record[categorical_cols]
    
    # Kết hợp kết quả
    return pd.concat([numeric_avg, categorical_nearest])

def merge_team_attributes(matches: pd.DataFrame, team_attributes: pd.DataFrame) -> pd.DataFrame:
    '''Gộp team attributes vào bảng trận đấu'''
    
    matches['home_team_attrs'] = matches.apply(
        lambda row: get_nearest_attributes(row['date'], row['home_team_api_id'], team_attributes),
        axis=1
    )
    matches['away_team_attrs'] = matches.apply(
        lambda row: get_nearest_attributes(row['date'], row['away_team_api_id'], team_attributes),
        axis=1
    )
    
    # Tách các cột từ home_team_attrs và away_team_attrs
    home_attrs = pd.DataFrame(matches['home_team_attrs'].tolist(), index=matches.index)
    home_attrs.columns = [f"{col}_home" for col in home_attrs.columns]
    
    away_attrs = pd.DataFrame(matches['away_team_attrs'].tolist(), index=matches.index)
    away_attrs.columns = [f"{col}_away" for col in away_attrs.columns]
    
    # Gộp vào dữ liệu chính
    matches = pd.concat([matches, home_attrs, away_attrs], axis=1)
    matches = matches.drop(['home_team_attrs', 'away_team_attrs'], axis=1)
    
    return matches

def calculate_home_win_rate(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''Tính tỷ lệ thắng sân nhà từ tập train'''
    
    home_wins = train[train['outcome'] == 'Win'].groupby('home_team_api_id').size()
    home_matches = train.groupby('home_team_api_id').size()
    home_win_rate = (home_wins / home_matches).fillna(0)
    
    train['home_win_rate'] = train['home_team_api_id'].map(home_win_rate).fillna(0)
    test['home_win_rate'] = test['home_team_api_id'].map(home_win_rate).fillna(0)
    print("Đã tính home_win_rate.")
    return train, test

def encode_categorical_columns(train: pd.DataFrame, test: pd.DataFrame, categorical_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''Mã hóa các cột nominal bằng One-Hot Encoding.'''
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Fit encoder trên tập train
    encoder.fit(train[categorical_cols])
    
    # Transform train
    train_encoded = pd.DataFrame(
        encoder.transform(train[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=train.index
    )
    train = pd.concat([train.drop(categorical_cols, axis=1), train_encoded], axis=1)
    
    # Transform test
    test_encoded = pd.DataFrame(
        encoder.transform(test[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=test.index
    )
    test = pd.concat([test.drop(categorical_cols, axis=1), test_encoded], axis=1)
    
    return train, test