import pandas as pd

def merge_tables(
    matches: pd.DataFrame,
    league: pd.DataFrame,
    country: pd.DataFrame,
    teams: pd.DataFrame,
) -> pd.DataFrame:

    # 1. Merge Match ↔ League ↔ Country
    df = matches.merge(league[['id','name']], left_on='league_id', right_on='id', how='left') \
    .rename(columns={'name':'league_name'}) \
    .merge(country[['id','name']], left_on='country_id', right_on='id', how='left') \
    .rename(columns={'name':'country_name'}) \
    .drop(columns=['id_x','id_y', 'id'])
    
    # 2. Merge Match ↔ Team
    teams_copy = teams.copy()
    teams_copy.rename(columns={'team_api_id': 'home_team_api_id', 'team_long_name': 'home_team_name'}, inplace = True)
    df = df.merge(teams_copy[['home_team_api_id', 'home_team_name']], on='home_team_api_id', how='left')
    
    teams_copy.rename(columns={'home_team_api_id': 'away_team_api_id', 'home_team_name': 'away_team_name'}, inplace = True)
    df = df.merge(teams_copy[['away_team_api_id', 'away_team_name']], on='away_team_api_id', how='left')
    
    return df