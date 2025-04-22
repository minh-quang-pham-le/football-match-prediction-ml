import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from .utils import check_missing_values

def load_raw_data(data_dir: str = 'data/raw'):
    '''Load dữ liệu gốc'''
    
    data_path = Path(data_dir)
    matches = pd.read_csv(data_path / 'match.csv')
    teams = pd.read_csv(data_path / 'teams.csv')
    team_attributes = pd.read_csv(data_path / 'team_attributes.csv')
    
    # Kiểm tra giá trị thiếu
    check_missing_values(matches, 'matches')
    check_missing_values(teams, 'teams')
    check_missing_values(team_attributes, 'team_attributes')
    
    return matches, teams, team_attributes

def convert_dates(matches: pd.DataFrame, team_attributes: pd.DataFrame):
    '''Chuyển đổi cột date sang định dạng datetime'''
    
    matches['date'] = pd.to_datetime(matches['date'])
    team_attributes['date'] = pd.to_datetime(team_attributes['date'])
    return matches, team_attributes

def create_match_outcome(matches: pd.DataFrame):
    '''Tạo target variable: Win/Loss/Draw'''
    
    matches['outcome'] = matches.apply(
        lambda row: 'Win' if row['home_team_goal'] > row['away_team_goal']
        else 'Draw' if row['home_team_goal'] == row['away_team_goal']
        else 'Loss',
        axis=1
    )
    return matches

def merge_team_names(matches: pd.DataFrame, teams: pd.DataFrame):
    '''Gộp tên đội chủ nhà và đội khách vào bảng trận đấu'''
    
    teams_copy = teams.copy()
    teams_copy.rename(columns={'team_api_id': 'home_team_api_id', 'team_long_name': 'home_team_name'}, inplace = True)
    matches = matches.merge(teams_copy[['home_team_api_id', 'home_team_name']], on='home_team_api_id', how='left')
    
    teams_copy.rename(columns={'home_team_api_id': 'away_team_api_id', 'home_team_name': 'away_team_name'}, inplace = True)
    matches = matches.merge(teams_copy[['away_team_api_id', 'away_team_name']], on='away_team_api_id', how='left')
    
    return matches

def split_train_test(matches, split_year=2015):
    train = matches[matches['date'].dt.year <= split_year]
    test = matches[matches['date'].dt.year > split_year]
    return train, test

def handle_missing_values(train: pd.DataFrame, test: pd.DataFrame):
    # Điền giá trị 0 cho cột số
    numeric_cols = train.select_dtypes(include=['float64', 'int64']).columns
    train[numeric_cols] = train[numeric_cols].fillna(0)
    test[numeric_cols] = test[numeric_cols].fillna(0)
    
    # Điền giá trị phổ biến nhất cho cột phân loại
    categorical_cols = train.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col not in ['home_team_name', 'away_team_name', 'outcome', 'date']:
            mode_value = train[col].mode()[0]
            train[col] = train[col].fillna(mode_value)
            test[col] = test[col].fillna(mode_value)
    
    return train, test