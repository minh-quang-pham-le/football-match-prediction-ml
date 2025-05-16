from data.data_preprocessing import *
from features.feature_engineering import *
from model.evaluate_model import *
from utils.utils import *
from model.train_model import *

def preprocess_data(data_dir='data/raw'):
    matches, teams, team_attributes = load_raw_data(data_dir)
    matches = create_match_outcome(matches)
    matches = merge_team_names(matches, teams)
    matches, team_attributes = convert_dates(matches, team_attributes)
    return matches, team_attributes

def feature_engineer_before_split(matches, team_attributes, split_year=2015):
    league_avg_goals = calculate_league_avg_goals(matches, split_year)
    matches = calculate_recent_goals(matches, league_avg_goals)
    matches = merge_team_attributes(matches, team_attributes)
    return matches

def split_data(matches, split_year=2015):
    train, test = split_train_test(matches, split_year)
    return train, test

def feature_engineer_after_split(train, test):
    train, test = calculate_home_win_rate(train, test)
    categorical_cols = [col for col in train.columns if col.endswith('Class_home') or col.endswith('Class_away')]
    if categorical_cols:
        train, test = encode_categorical_columns(train, test, categorical_cols)
    train, test = handle_missing_values(train, test)
    return train, test

def process_football_data(data_dir='data/raw', split_year=2015):
    # Bước 1: Tiền xử lý
    matches, team_attributes = preprocess_data(data_dir)

    # Bước 2: Feature engineering trước khi chia
    matches = feature_engineer_before_split(matches, team_attributes, split_year)

    # Bước 3: Chia dữ liệu
    train, test = split_data(matches, split_year)

    # Bước 4: Feature engineering sau khi chia
    train, test = feature_engineer_after_split(train, test)

    return train, test

if __name__ == "__main__":
    train_df, test_df = process_football_data(data_dir='data/raw', split_year=2015)