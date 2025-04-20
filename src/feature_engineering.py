import pandas as pd

def build_features(filepath):
    df = pd.read_csv(filepath)
    # Ví dụ: hiệu suất đội nhà, đội khách
    df['goal_diff'] = df['home_team_goal'] - df['away_team_goal']
    return df
