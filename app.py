import os
import pandas as pd
import numpy as np
import gradio as gr
from gradio import update  # Add this import
import joblib
import json
from datetime import datetime
from typing import List, Dict, Tuple
from src.data.preprocess_data import recalc_diff, outcome_to_int

# Paths
MODELS_DIR = 'models'
DATA_DIR = 'data'

# Load necessary data
def load_data():
    """Load teams, players, and model data for the app"""
    # Load test data from X_test.csv instead of relying on Team.csv
    X_test_df = pd.read_csv(os.path.join(DATA_DIR, 'feature', 'X_test.csv'))
    
    # Create teams_df with unique team IDs from X_test.csv
    home_teams = X_test_df[['home_team_api_id']].rename(columns={'home_team_api_id': 'team_api_id'}).drop_duplicates()
    away_teams = X_test_df[['away_team_api_id']].rename(columns={'away_team_api_id': 'team_api_id'}).drop_duplicates()
    team_ids = pd.concat([home_teams, away_teams]).drop_duplicates()
    
    # Load teams mapping if available, otherwise create a default mapping
    try:
        teams_df = pd.read_csv(os.path.join(DATA_DIR, 'raw', 'Team.csv'))
        # Filter to only include teams in X_test
        teams_df = teams_df[teams_df['team_api_id'].isin(team_ids['team_api_id'])]
    except Exception as e:
        print(f"Could not load Team.csv: {e}")
        # Create a fallback teams dataframe with generic team names
        teams_df = team_ids.copy()
        teams_df['team_long_name'] = 'Team ' + teams_df['team_api_id'].astype(str)
        teams_df['team_short_name'] = 'T' + teams_df['team_api_id'].astype(str)
    
    # Load players
    players_df = pd.read_csv(os.path.join(DATA_DIR, 'raw', 'Player.csv'))
    player_attrs_df = pd.read_csv(os.path.join(DATA_DIR, 'raw', 'Player_Attributes.csv'))
    
    # Load team attributes
    team_attrs_df = pd.read_csv(os.path.join(DATA_DIR, 'raw', 'Team_Attributes.csv'))
    
    # Load matches (we'll need this for extracting test set data)
    matches_df = pd.read_csv(os.path.join(DATA_DIR, 'raw', 'Match.csv'))
    
    # Load processed data
    try:
        processed_df = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'df_2.csv'))
        print(f"Successfully loaded processed data with {len(processed_df)} rows")
    except Exception as e:
        print(f"Could not load processed data: {e}")
        processed_df = pd.DataFrame()
    
    # Filter to include only test season (2015/2016)
    test_season = '2015/2016'
    test_matches_df = matches_df[matches_df['season'] == test_season].copy()
    
    # Get all available models
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    models = {f.replace('_best.pkl', ''): f for f in model_files if '_best.pkl' in f}
    
    return teams_df, players_df, player_attrs_df, team_attrs_df, test_matches_df, processed_df, models

# Load model and associated metadata
def load_model(model_name):
    """Load model and its metadata by name"""
    model_path = os.path.join(MODELS_DIR, f"{model_name}_best.pkl")
    metadata_path = os.path.join(MODELS_DIR, f"{model_name}_best_metadata.json")
    
    model = joblib.load(model_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return model, metadata

# Get players for a specific team
def get_players_for_team(team_api_id, players_df, player_attrs_df, test_matches_df, processed_df=None):
    """Get players who have played for the team in the test season"""
    print(f"DEBUG: Getting players for team_api_id {team_api_id}")
    
    # Find matches for this team
    team_matches = test_matches_df[
        (test_matches_df['home_team_api_id'] == team_api_id) |
        (test_matches_df['away_team_api_id'] == team_api_id)
    ]
    
    print(f"DEBUG: Found {len(team_matches)} matches for team {team_api_id}")
    
    # Extract all player IDs for this team from the matches
    player_ids = set()
    for _, match in team_matches.iterrows():
        if match['home_team_api_id'] == team_api_id:
            for i in range(1, 12):
                player_id = match[f'home_player_{i}']
                if pd.notna(player_id):
                    player_ids.add(int(player_id))
        else:
            for i in range(1, 12):
                player_id = match[f'away_player_{i}']
                if pd.notna(player_id):
                    player_ids.add(int(player_id))
    
    print(f"DEBUG: Extracted {len(player_ids)} unique player IDs for team {team_api_id}")
    
    # Get player information
    team_players = players_df[players_df['player_api_id'].isin(player_ids)].copy()
    print(f"DEBUG: Found {len(team_players)} players in players_df for team {team_api_id}")
    
    # Get latest attributes for each player
    latest_attrs = player_attrs_df.sort_values('date', ascending=False).drop_duplicates('player_api_id')
    
    # Merge with player attributes
    team_players = team_players.merge(
        latest_attrs[['player_api_id', 'overall_rating', 'potential']], 
        on='player_api_id', 
        how='left'
    )
    
    print(f"DEBUG: After merge with attributes, have {len(team_players)} players for team {team_api_id}")
    
    # Format player display names
    team_players['display_name'] = team_players.apply(
        lambda x: f"{x['player_name']} ({int(x['overall_rating']) if pd.notna(x['overall_rating']) else 'NA'})", 
        axis=1
    )
    
    # Create a dict for dropdown {display_name: player_api_id}
    player_dict = dict(zip(team_players['display_name'], team_players['player_api_id']))
    
    print(f"DEBUG: Generated player dict for team {team_api_id}: {len(player_dict)} players found")
    if not player_dict:
        print(f"WARNING: No players found for team {team_api_id}!")
    # Debug: Print player dict for verification
    print(f"Generated player dict for team {team_api_id}: {len(player_dict)} players found")
    if not player_dict:
        print("WARNING: No players found for this team!")
    
    return player_dict

# Generate features for prediction
def generate_features(
    home_team_id, away_team_id, 
    home_players, away_players,
    selected_date,
    teams_df, players_df, player_attrs_df, team_attrs_df, test_matches_df
):
    """Generate all features needed for model prediction"""
    
    # 1. Create a dataframe with one row for prediction
    pred_df = pd.DataFrame({
        'league_id': [test_matches_df['league_id'].iloc[0]],  # Use league from test set
        'match_api_id': [999999],  # dummy value
        'season': ['2015/2016'],  # Test season
        'prev_season': ['2014/2015'],  # Previous season
        'date': [selected_date],
        'home_team_api_id': [home_team_id],
        'away_team_api_id': [away_team_id],
        'stage': [test_matches_df['stage'].max()],  # Use last stage from test set
    })
    
    # 2. Add player IDs
    for i, player_id in enumerate(home_players, 1):
        pred_df[f'home_player_{i}'] = player_id
    
    for i, player_id in enumerate(away_players, 1):
        pred_df[f'away_player_{i}'] = player_id
    
    # 3. Calculate match_year and match_month
    pred_df['match_year'] = pd.to_datetime(pred_df['date']).dt.year
    pred_df['match_month'] = pd.to_datetime(pred_df['date']).dt.month
    
    # 4. Calculate team form features (wins, goals scored/conceded in last 5 matches)
    # Get past matches for both teams
    home_past_matches = test_matches_df[
        ((test_matches_df['home_team_api_id'] == home_team_id) | 
        (test_matches_df['away_team_api_id'] == home_team_id)) &
        (pd.to_datetime(test_matches_df['date']) < pd.to_datetime(selected_date))
    ].sort_values('date', ascending=False).head(5)
    
    away_past_matches = test_matches_df[
        ((test_matches_df['home_team_api_id'] == away_team_id) | 
        (test_matches_df['away_team_api_id'] == away_team_id)) &
        (pd.to_datetime(test_matches_df['date']) < pd.to_datetime(selected_date))
    ].sort_values('date', ascending=False).head(5)
    
    # Calculate form metrics with weighted averages (more weight to recent matches)
    weights = np.array([5, 4, 3, 2, 1])  # Higher weight for recent matches
    
    # Home team form
    home_wins = []
    home_goals_for = []
    home_goals_against = []
    
    for _, match in home_past_matches.iterrows():
        if match['home_team_api_id'] == home_team_id:
            home_wins.append(1 if match['home_team_goal'] > match['away_team_goal'] else 0)
            home_goals_for.append(match['home_team_goal'])
            home_goals_against.append(match['away_team_goal'])
        else:
            home_wins.append(1 if match['away_team_goal'] > match['home_team_goal'] else 0)
            home_goals_for.append(match['away_team_goal'])
            home_goals_against.append(match['home_team_goal'])
    
    # Away team form
    away_wins = []
    away_goals_for = []
    away_goals_against = []
    
    for _, match in away_past_matches.iterrows():
        if match['home_team_api_id'] == away_team_id:
            away_wins.append(1 if match['home_team_goal'] > match['away_team_goal'] else 0)
            away_goals_for.append(match['home_team_goal'])
            away_goals_against.append(match['away_team_goal'])
        else:
            away_wins.append(1 if match['away_team_goal'] > match['home_team_goal'] else 0)
            away_goals_for.append(match['away_team_goal'])
            away_goals_against.append(match['home_team_goal'])
    
    # Calculate weighted averages
    used_weights = weights[:len(home_wins)]
    pred_df['home_wins_last5'] = np.average(home_wins, weights=used_weights) if home_wins else 0
    pred_df['home_avg_gs_last5'] = np.average(home_goals_for, weights=used_weights) if home_goals_for else 0
    pred_df['home_avg_gc_last5'] = np.average(home_goals_against, weights=used_weights) if home_goals_against else 0
    
    used_weights = weights[:len(away_wins)]
    pred_df['away_wins_last5'] = np.average(away_wins, weights=used_weights) if away_wins else 0
    pred_df['away_avg_gs_last5'] = np.average(away_goals_for, weights=used_weights) if away_goals_for else 0
    pred_df['away_avg_gc_last5'] = np.average(away_goals_against, weights=used_weights) if away_goals_against else 0
    
    # 5. Head-to-head features
    h2h_matches = test_matches_df[
        ((test_matches_df['home_team_api_id'] == home_team_id) & (test_matches_df['away_team_api_id'] == away_team_id) |
         (test_matches_df['home_team_api_id'] == away_team_id) & (test_matches_df['away_team_api_id'] == home_team_id)) &
        (pd.to_datetime(test_matches_df['date']) < pd.to_datetime(selected_date))
    ].sort_values('date', ascending=False).head(3)
    
    h2h_wins = 0
    for _, match in h2h_matches.iterrows():
        if match['home_team_api_id'] == home_team_id:
            h2h_wins += 1 if match['home_team_goal'] > match['away_team_goal'] else 0
        else:
            h2h_wins += 1 if match['away_team_goal'] > match['home_team_goal'] else 0
    
    pred_df['h2h_wins_last3'] = h2h_wins
    pred_df['h2h_nobs'] = len(h2h_matches)
    
    # Apply Laplace smoothing
    alpha = 2
    p0 = 0.5
    pred_df['h2h_win_rate_last3'] = (h2h_wins + alpha * p0) / (len(h2h_matches) + alpha)
    
    # 6. Team tactics features
    home_tactics = team_attrs_df[team_attrs_df['team_api_id'] == home_team_id].sort_values('date', ascending=False).iloc[0]
    away_tactics = team_attrs_df[team_attrs_df['team_api_id'] == away_team_id].sort_values('date', ascending=False).iloc[0]
    
    tactics_attrs = [
        'buildUpPlaySpeed', 'buildUpPlayPassing',
        'chanceCreationPassing', 'chanceCreationShooting',
        'defencePressure', 'defenceAggression', 'defenceTeamWidth'
    ]
    
    for attr in tactics_attrs:
        pred_df[f'home_{attr}'] = home_tactics[attr] if attr in home_tactics else None
        pred_df[f'away_{attr}'] = away_tactics[attr] if attr in away_tactics else None
    
    # 7. Tactical differences
    pred_df['diff_speed'] = pred_df['home_buildUpPlaySpeed'] - pred_df['away_buildUpPlaySpeed']
    pred_df['diff_shooting'] = pred_df['home_chanceCreationShooting'] - pred_df['away_chanceCreationShooting']
    pred_df['diff_pressure'] = pred_df['home_defencePressure'] - pred_df['away_defencePressure']
    
    # 8. Player attributes
    # Get latest player attributes
    selected_players = home_players + away_players
    player_attributes = player_attrs_df[player_attrs_df['player_api_id'].isin(selected_players)].sort_values('date', ascending=False).drop_duplicates('player_api_id')
    
    # Calculate average physical attributes for home and away teams
    home_players_attrs = player_attributes[player_attributes['player_api_id'].isin(home_players)]
    away_players_attrs = player_attributes[player_attributes['player_api_id'].isin(away_players)]
    
    # Physical attributes
    pred_df['home_avg_height'] = home_players_attrs['height'].mean()
    pred_df['away_avg_height'] = away_players_attrs['height'].mean()
    pred_df['home_avg_weight'] = home_players_attrs['weight'].mean()
    pred_df['away_avg_weight'] = away_players_attrs['weight'].mean()
    
    # Calculate average age
    player_info = players_df[players_df['player_api_id'].isin(selected_players)]
    
    # Convert birthday to datetime
    player_info['birthday'] = pd.to_datetime(player_info['birthday'])
    selected_date_dt = pd.to_datetime(selected_date)
    
    # Merge player info with home and away players
    home_players_info = player_info[player_info['player_api_id'].isin(home_players)]
    away_players_info = player_info[player_info['player_api_id'].isin(away_players)]
    
    # Calculate age for each player
    home_players_info['age'] = (selected_date_dt - home_players_info['birthday']).dt.days / 365
    away_players_info['age'] = (selected_date_dt - away_players_info['birthday']).dt.days / 365
    
    # Average age
    pred_df['home_avg_age'] = home_players_info['age'].mean()
    pred_df['away_avg_age'] = away_players_info['age'].mean()
    
    # 9. Player skills
    # Calculate average ratings for different skills
    for side, team_attrs in [('home', home_players_attrs), ('away', away_players_attrs)]:
        pred_df[f'{side}_avg_overall_rating'] = team_attrs['overall_rating'].mean()
        pred_df[f'{side}_avg_potential'] = team_attrs['potential'].mean()
        pred_df[f'{side}_avg_pace'] = team_attrs[['acceleration', 'sprint_speed']].mean(axis=1).mean()
        pred_df[f'{side}_avg_passing_skill'] = team_attrs[['short_passing', 'long_passing']].mean(axis=1).mean()
        pred_df[f'{side}_avg_dribbling_skill'] = team_attrs['dribbling'].mean()
        pred_df[f'{side}_avg_shooting_skill'] = team_attrs[['shot_power', 'long_shots']].mean(axis=1).mean()
        pred_df[f'{side}_avg_finishing'] = team_attrs['finishing'].mean()
        pred_df[f'{side}_avg_physical'] = team_attrs[['stamina', 'strength']].mean(axis=1).mean()
        pred_df[f'{side}_avg_defensive_skill'] = team_attrs[['standing_tackle', 'sliding_tackle']].mean(axis=1).mean()
        pred_df[f'{side}_avg_crossing'] = team_attrs['crossing'].mean()
        pred_df[f'{side}_avg_heading_accuracy'] = team_attrs['heading_accuracy'].mean()
        pred_df[f'{side}_avg_penalties'] = team_attrs['penalties'].mean()
    
    # 10. Match details
    pred_df['match_importance'] = 1.0  # Assuming maximum importance
    pred_df['match_phase'] = 3  # Late phase (3rd out of 3)
    
    # 11. Add betting odds (use median values from test set as fallback)
    odds_cols = [c for c in test_matches_df if c.startswith(('B365','BW','LB','WH','VC','IW'))]
    for col in odds_cols:
        pred_df[col] = test_matches_df[col].median()
    
    return pred_df

# Predict using processed data
def predict_with_processed_data(model, home_team_id, away_team_id, match_date, processed_df, metadata):
    """Make prediction using processed data if available"""
    if processed_df.empty:
        return None
    
    # Find matches in processed data with these teams
    team_matches = processed_df[
        ((processed_df['home_team_api_id'] == home_team_id) & (processed_df['away_team_api_id'] == away_team_id)) |
        ((processed_df['home_team_api_id'] == away_team_id) & (processed_df['away_team_api_id'] == home_team_id))
    ]
    
    if len(team_matches) == 0:
        print(f"No matches found for teams {home_team_id} vs {away_team_id} in processed data")
        return None
    
    # Use the latest match as a template and update date
    template_match = team_matches.sort_values('date', ascending=False).iloc[0:1].copy()
    template_match['date'] = match_date
    
    # Ensure home and away are in correct order
    if template_match['home_team_api_id'].iloc[0] != home_team_id:
        # Swap home and away features
        cols_to_swap = {col: col.replace('home_', 'away_') if 'home_' in col else 
                       (col.replace('away_', 'home_') if 'away_' in col else col) 
                      for col in template_match.columns}
        template_match = template_match.rename(columns=cols_to_swap)
        
        # Fix team IDs
        template_match['home_team_api_id'] = home_team_id
        template_match['away_team_api_id'] = away_team_id
        
        # Recalculate outcome if needed
        if 'outcome' in template_match.columns:
            template_match['outcome'] = template_match['outcome'].map({0: 0, 1: 1, 2: 2})
    
    # Ensure we have all required features
    required_features = metadata['features']
    
    # Create missing columns with NaN values
    missing_features = [feature for feature in required_features if feature not in template_match.columns]
    if missing_features:
        print(f"Missing features in processed data: {missing_features}")
        return None
    
    # Select only the features required by the model
    X = template_match[required_features]
    
    # Predict probabilities
    y_proba = model.predict_proba(X)
    
    return {
        'Loss': y_proba[0][0],
        'Draw': y_proba[0][1],
        'Win': y_proba[0][2]
    }

# Predict match outcome
def predict_match(model, features_df, metadata):
    """Make prediction with the selected model"""
    # Ensure we have all required features
    required_features = metadata['features']
    
    # Create missing columns with NaN values
    for feature in required_features:
        if feature not in features_df.columns:
            features_df[feature] = np.nan
    
    # Select only the features required by the model
    X = features_df[required_features]
    
    # Predict probabilities
    y_proba = model.predict_proba(X)
    
    # Map back to outcomes
    return {
        'Loss': y_proba[0][0],
        'Draw': y_proba[0][1],
        'Win': y_proba[0][2]
    }

# Main Gradio interface function
def predict_football_match(
    home_team, away_team,
    home_player_1, home_player_2, home_player_3, home_player_4, home_player_5, 
    home_player_6, home_player_7, home_player_8, home_player_9, home_player_10, home_player_11,
    away_player_1, away_player_2, away_player_3, away_player_4, away_player_5,
    away_player_6, away_player_7, away_player_8, away_player_9, away_player_10, away_player_11,
    match_date, selected_model
):
    try:
        # Extract team IDs - if the input is a tuple like (name, id), use the id (second item)
        if isinstance(home_team, tuple):
            home_team_id = int(home_team[1])
        else:
            home_team_id = int(home_team)
            
        if isinstance(away_team, tuple):
            away_team_id = int(away_team[1])
        else:
            away_team_id = int(away_team)
        
        # Extract model name if it's a tuple
        if isinstance(selected_model, tuple):
            selected_model = selected_model[1]
        
        # Get player IDs (they're passed from the dropdown value)
        home_players = [
            home_player_1, home_player_2, home_player_3, home_player_4, home_player_5,
            home_player_6, home_player_7, home_player_8, home_player_9, home_player_10, home_player_11
        ]
        
        away_players = [
            away_player_1, away_player_2, away_player_3, away_player_4, away_player_5,
            away_player_6, away_player_7, away_player_8, away_player_9, away_player_10, away_player_11
        ]
        
        # Load data
        teams_df, players_df, player_attrs_df, team_attrs_df, test_matches_df, processed_df, models = load_data()
        
        # Load selected model
        model, metadata = load_model(selected_model)
        
        # Try to predict using processed data first
        prediction = None
        if not processed_df.empty:
            print("Attempting to use processed data for prediction...")
            prediction = predict_with_processed_data(
                model, home_team_id, away_team_id, match_date, processed_df, metadata
            )
        
        # If processed data prediction failed, fall back to feature generation
        if prediction is None:
            print("Falling back to manual feature generation...")
            # Generate features
            features_df = generate_features(
                home_team_id, away_team_id, 
                home_players, away_players, 
                match_date,
                teams_df, players_df, player_attrs_df, team_attrs_df, test_matches_df
            )
            
            # Make prediction
            prediction = predict_match(model, features_df, metadata)
            
        # Prepare result
        # Get team names with safer handling in case team is not found
        home_team_filter = teams_df['team_api_id'] == home_team_id
        away_team_filter = teams_df['team_api_id'] == away_team_id
        
        home_team_name = teams_df[home_team_filter]['team_long_name'].iloc[0] if any(home_team_filter) else f"Team {home_team_id}"
        away_team_name = teams_df[away_team_filter]['team_long_name'].iloc[0] if any(away_team_filter) else f"Team {away_team_id}"
        
        # Format the output
        result = f"## Match Prediction: {home_team_name} vs {away_team_name}\n\n"
        result += f"**Win Probability for {home_team_name}:** {prediction['Win']*100:.2f}%\n\n"
        result += f"**Draw Probability:** {prediction['Draw']*100:.2f}%\n\n"
        result += f"**Win Probability for {away_team_name}:** {prediction['Loss']*100:.2f}%\n\n"
        result += f"**Model Used:** {selected_model}\n"
        
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

# UI update functions
def update_home_players(home_team):
    """Update home player dropdown options when team is selected"""
    teams_df, players_df, player_attrs_df, team_attrs_df, test_matches_df, processed_df, models = load_data()
    
    # Extract team ID if it's a tuple
    if isinstance(home_team, tuple):
        home_team_id = int(home_team[1])
    else:
        home_team_id = int(home_team)
        
    player_dict = get_players_for_team(home_team_id, players_df, player_attrs_df, test_matches_df, processed_df)
    # Return 11 identical dropdowns - use gr.update() instead of gr.Dropdown.update()
    return [gr.update(choices=list(player_dict.keys()), value=None) for _ in range(11)]

def update_away_players(away_team):
    """Update away player dropdown options when team is selected"""
    teams_df, players_df, player_attrs_df, team_attrs_df, test_matches_df, processed_df, models = load_data()
    
    # Extract team ID if it's a tuple
    if isinstance(away_team, tuple):
        away_team_id = int(away_team[1])
    else:
        away_team_id = int(away_team)
        
    player_dict = get_players_for_team(away_team_id, players_df, player_attrs_df, test_matches_df, processed_df)
    # Return 11 identical dropdowns - use gr.update() instead of gr.Dropdown.update()
    return [gr.update(choices=list(player_dict.keys()), value=None) for _ in range(11)]

def load_teams():
    """Load teams for dropdowns using data from X_test.csv"""
    teams_df, _, _, _, _, processed_df, _ = load_data()
    
    # Use teams directly from the teams_df created in load_data()    # (which now contains only teams from X_test.csv)
    available_teams = teams_df
    print(f"Found {len(available_teams)} teams in X_test data")
    
    # Create a dict mapping team name to team_api_id
    team_dict = dict(zip(available_teams['team_long_name'], available_teams['team_api_id']))
    return gr.update(choices=list(team_dict.items()))

def load_models():
    """Load available models for dropdown"""
    _, _, _, _, _, _, models = load_data()
    
    # Check if models are available
    if not models:
        print("Warning: No models found in models directory")
        return gr.update(choices=[])
    
    # Return model names with descriptions for better display
    model_choices = [(f"{name} Model", name) for name in models.keys()]
    return gr.update(choices=model_choices)

def player_selection_to_id(selection, team_id):
    """Convert player display name to player_api_id"""
    teams_df, players_df, player_attrs_df, team_attrs_df, test_matches_df, processed_df, _ = load_data()
    
    # Extract team ID if it's a tuple
    if isinstance(team_id, tuple):
        team_id = int(team_id[1])
    else:
        team_id = int(team_id)
        
    player_dict = get_players_for_team(team_id, players_df, player_attrs_df, test_matches_df, processed_df)
    return player_dict.get(selection)

# Create Gradio Interface
def create_interface():
    with gr.Blocks(title="Football Match Prediction") as app:
        gr.Markdown("# Football Match Prediction")
        gr.Markdown("Select two teams, their players, and a match date to predict the outcome using machine learning.")
        
        # Load team and model data first - this gets the data ready for the dropdowns
        teams_df, players_df, player_attrs_df, team_attrs_df, test_matches_df, processed_df, models_dict = load_data()
        team_choices = [(name, id) for name, id in zip(teams_df['team_long_name'], teams_df['team_api_id'])]
        model_choices = [(f"{name} Model", name) for name in models_dict.keys()]
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Home Team")
                home_team = gr.Dropdown(label="Select Home Team", choices=team_choices, interactive=True)
                
                gr.Markdown("#### Select 11 Players")
                home_player_1 = gr.Dropdown(label="Player 1")
                home_player_2 = gr.Dropdown(label="Player 2")
                home_player_3 = gr.Dropdown(label="Player 3")
                home_player_4 = gr.Dropdown(label="Player 4")
                home_player_5 = gr.Dropdown(label="Player 5")
                home_player_6 = gr.Dropdown(label="Player 6")
                home_player_7 = gr.Dropdown(label="Player 7")
                home_player_8 = gr.Dropdown(label="Player 8")
                home_player_9 = gr.Dropdown(label="Player 9")
                home_player_10 = gr.Dropdown(label="Player 10")
                home_player_11 = gr.Dropdown(label="Player 11")
                
            with gr.Column():
                gr.Markdown("### Away Team")
                away_team = gr.Dropdown(label="Select Away Team", choices=team_choices, interactive=True)
                
                gr.Markdown("#### Select 11 Players")
                away_player_1 = gr.Dropdown(label="Player 1")
                away_player_2 = gr.Dropdown(label="Player 2")
                away_player_3 = gr.Dropdown(label="Player 3")
                away_player_4 = gr.Dropdown(label="Player 4")
                away_player_5 = gr.Dropdown(label="Player 5")
                away_player_6 = gr.Dropdown(label="Player 6")
                away_player_7 = gr.Dropdown(label="Player 7")
                away_player_8 = gr.Dropdown(label="Player 8")
                away_player_9 = gr.Dropdown(label="Player 9")
                away_player_10 = gr.Dropdown(label="Player 10")
                away_player_11 = gr.Dropdown(label="Player 11")
        
        with gr.Row():
            match_date = gr.Textbox(
                label="Match Date (YYYY-MM-DD)", 
                value=datetime.now().strftime("%Y-%m-%d")
            )
            model_selector = gr.Dropdown(label="Select Prediction Model", choices=model_choices)
        
        predict_btn = gr.Button("Predict Match Outcome")
        result_output = gr.Markdown()
        
        # Update player dropdowns when team is selected
        home_team.change(
            fn=update_home_players,
            inputs=home_team,
            outputs=[
                home_player_1, home_player_2, home_player_3, home_player_4, home_player_5,
                home_player_6, home_player_7, home_player_8, home_player_9, home_player_10, home_player_11
            ]
        )
        
        away_team.change(
            fn=update_away_players,
            inputs=away_team,
            outputs=[
                away_player_1, away_player_2, away_player_3, away_player_4, away_player_5,
                away_player_6, away_player_7, away_player_8, away_player_9, away_player_10, away_player_11
            ]
        )
        
        # Run prediction when button clicked
        predict_btn.click(
            fn=predict_football_match,
            inputs=[
                home_team, away_team,
                home_player_1, home_player_2, home_player_3, home_player_4, home_player_5,
                home_player_6, home_player_7, home_player_8, home_player_9, home_player_10, home_player_11,
                away_player_1, away_player_2, away_player_3, away_player_4, away_player_5,
                away_player_6, away_player_7, away_player_8, away_player_9, away_player_10, away_player_11,
                match_date, model_selector
            ],
            outputs=result_output
        )
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch()
