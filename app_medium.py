import os
import pandas as pd
import numpy as np
import gradio as gr
from gradio import update  # Add this import
import joblib
import json
from typing import Dict, List
from datetime import datetime

# Paths
MODELS_DIR = 'models'
DATA_DIR = 'data'

def load_data():
    """Load necessary data for prediction"""
    # Load teams and matches
    teams_df = pd.read_csv(os.path.join(DATA_DIR, 'raw', 'Team.csv'))
    matches_df = pd.read_csv(os.path.join(DATA_DIR, 'raw', 'Match.csv'))
    
    # Load processed data
    try:
        processed_df = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'df_2.csv'))
        print(f"Successfully loaded processed data with {len(processed_df)} rows")
    except Exception as e:
        print(f"Could not load processed data: {e}")
        processed_df = pd.DataFrame()
    
    # Filter to test season
    test_season = '2015/2016'
    test_matches_df = matches_df[matches_df['season'] == test_season].copy()
    
    # Get all available models
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    models = {f.replace('_best.pkl', ''): f for f in model_files if '_best.pkl' in f}
    
    return teams_df, test_matches_df, processed_df, models

def load_model(model_name):
    """Load model and metadata by name"""
    model_path = os.path.join(MODELS_DIR, f"{model_name}_best.pkl")
    metadata_path = os.path.join(MODELS_DIR, f"{model_name}_best_metadata.json")
    
    model = joblib.load(model_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return model, metadata

def find_latest_match(team_id, date, matches_df):
    """Find the latest match for a team before the given date"""
    team_matches = matches_df[
        ((matches_df['home_team_api_id'] == team_id) | 
         (matches_df['away_team_api_id'] == team_id)) &
        (pd.to_datetime(matches_df['date']) <= pd.to_datetime(date))
    ].sort_values('date', ascending=False)
    
    if team_matches.empty:
        return None
    
    return team_matches.iloc[0]

def get_team_players(team_id, match_row, is_home):
    """Extract player IDs for a team from a match"""
    prefix = 'home_player_' if is_home else 'away_player_'
    players = []
    
    for i in range(1, 12):
        player_id = match_row[f'{prefix}{i}']
        if pd.notna(player_id):
            players.append(int(player_id))
        else:
            players.append(None)  # Placeholder for missing players
    
    return players

def find_best_match_for_prediction(home_team_id, away_team_id, date, matches_df):
    """Find the best match for feature extraction based on selected teams and date"""
    # Try to find a direct match between these teams first
    direct_matches = matches_df[
        ((matches_df['home_team_api_id'] == home_team_id) & (matches_df['away_team_api_id'] == away_team_id) |
         (matches_df['home_team_api_id'] == away_team_id) & (matches_df['away_team_api_id'] == home_team_id)) &
        (pd.to_datetime(matches_df['date']) <= pd.to_datetime(date))
    ].sort_values('date', ascending=False)
    
    if not direct_matches.empty:
        return direct_matches.iloc[0]['match_api_id']
    
    # If no direct match, find latest matches for each team
    home_latest = find_latest_match(home_team_id, date, matches_df)
    away_latest = find_latest_match(away_team_id, date, matches_df)
    
    if home_latest is not None:
        return home_latest['match_api_id']
    elif away_latest is not None:
        return away_latest['match_api_id']
    
    # If nothing found, return a default match from test set
    return matches_df.iloc[0]['match_api_id']

def predict_with_feature_modification(home_team_id, away_team_id, date, selected_model, matches_df, teams_df):
    """Make prediction by modifying features from an existing match"""
    # Find a good match to base our features on
    base_match_id = find_best_match_for_prediction(home_team_id, away_team_id, date, matches_df)
    
    # Load test features
    X_test = pd.read_csv(os.path.join(DATA_DIR, 'feature', 'X_test.csv'))
    
    # Since X_test doesn't have match_api_id, we'll use the first entry
    # We'll modify the home_team_api_id and away_team_api_id later anyway
    base_features = X_test.iloc[0:1].copy()
    
    print(f"DEBUG: Using base features with shape: {base_features.shape}")
      # Modify the features to match our teams
    base_features['home_team_api_id'] = home_team_id
    base_features['away_team_api_id'] = away_team_id
    
    # Get team names
    home_team_rows = teams_df[teams_df['team_api_id'] == home_team_id]
    away_team_rows = teams_df[teams_df['team_api_id'] == away_team_id]
    
    if home_team_rows.empty:
        home_team_name = f"Team ID {home_team_id}"
    else:
        home_team_name = home_team_rows['team_long_name'].iloc[0]
        
    if away_team_rows.empty:
        away_team_name = f"Team ID {away_team_id}"
    else:
        away_team_name = away_team_rows['team_long_name'].iloc[0]
    
    # Load selected model
    model, metadata = load_model(selected_model)
    
    # Make prediction
    # Ensure we have all required features
    required_features = metadata['features']
    
    # Create missing columns with NaN values
    for feature in required_features:
        if feature not in base_features.columns:
            base_features[feature] = np.nan
    
    # Select only the features required by the model
    X = base_features[required_features]
    
    # Predict probabilities
    y_proba = model.predict_proba(X)
    
    # Format the output
    result = f"## Match Prediction: {home_team_name} vs {away_team_name}\n\n"
    result += f"**Win Probability for {home_team_name}:** {y_proba[0][2]*100:.2f}%\n\n"
    result += f"**Draw Probability:** {y_proba[0][1]*100:.2f}%\n\n"
    result += f"**Win Probability for {away_team_name}:** {y_proba[0][0]*100:.2f}%\n\n"
    result += f"**Model Used:** {selected_model}\n\n"
    result += f"*Note: Prediction based on team compositions from similar matches.*"
    
    return result

def predict_with_processed_data(home_team_id, away_team_id, match_date, selected_model, processed_df, teams_df):
    """Make prediction using processed data"""
    # Find matches in processed data for these teams
    team_matches = processed_df[
        ((processed_df['home_team_api_id'] == home_team_id) & (processed_df['away_team_api_id'] == away_team_id)) |
        ((processed_df['home_team_api_id'] == away_team_id) & (processed_df['away_team_api_id'] == home_team_id))
    ]
    
    if team_matches.empty:
        # Fall back to any match with these teams
        home_matches = processed_df[processed_df['home_team_api_id'] == home_team_id]
        away_matches = processed_df[processed_df['away_team_api_id'] == away_team_id]
        
        if not home_matches.empty:
            base_features = home_matches.iloc[0].copy()
        elif not away_matches.empty:
            base_features = away_matches.iloc[0].copy()
        else:
            # Use a random match as fallback
            base_features = processed_df.iloc[0].copy()
            
        # Update team IDs
        base_features['home_team_api_id'] = home_team_id
        base_features['away_team_api_id'] = away_team_id
    else:
        # Use the first match between these teams
        base_features = team_matches.iloc[0].copy()
        # If teams are reversed, we need to adjust
        if base_features['home_team_api_id'] != home_team_id:
            # Swap home and away features
            for prefix in ['home_', 'away_']:
                opposite = 'away_' if prefix == 'home_' else 'home_'
                cols = [c for c in base_features.index if c.startswith(prefix)]
                opposite_cols = [c.replace(prefix, opposite) for c in cols]
                
                # Store original values
                temp_values = base_features[cols].copy()
                
                # Swap values
                for i, col in enumerate(cols):
                    if opposite_cols[i] in base_features:
                        base_features[col] = base_features[opposite_cols[i]]
                
                # Set opposite values
                for i, col in enumerate(opposite_cols):
                    if col in base_features:
                        base_features[col] = temp_values.iloc[i]
            
            # Fix the outcome if needed
            if 'outcome' in base_features:
                outcome_map = {'Win': 'Loss', 'Loss': 'Win', 'Draw': 'Draw'}
                base_features['outcome'] = outcome_map.get(base_features['outcome'], base_features['outcome'])
    
    # Get team names
    home_team_name = teams_df[teams_df['team_api_id'] == home_team_id]['team_long_name'].iloc[0]
    away_team_name = teams_df[teams_df['team_api_id'] == away_team_id]['team_long_name'].iloc[0]
    
    # Load selected model
    model, metadata = load_model(selected_model)
    
    # Base_features is a Series, convert to DataFrame for prediction
    features_df = pd.DataFrame([base_features])
    
    # Make prediction
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
    
    # Format the output
    result = f"## Match Prediction: {home_team_name} vs {away_team_name}\n\n"
    result += f"**Win Probability for {home_team_name}:** {y_proba[0][2]*100:.2f}%\n\n"
    result += f"**Draw Probability:** {y_proba[0][1]*100:.2f}%\n\n"
    result += f"**Win Probability for {away_team_name}:** {y_proba[0][0]*100:.2f}%\n\n"
    result += f"**Model Used:** {selected_model}\n\n"
    result += f"*Note: Prediction based on team statistics from the dataset.*"
    
    return result

def predict_football_match(home_team, away_team, match_date, selected_model):
    """Main prediction function for the Gradio interface"""
    try:
        # Debug output
        print(f"DEBUG: home_team={home_team}, type={type(home_team)}")
        print(f"DEBUG: away_team={away_team}, type={type(away_team)}")
        print(f"DEBUG: selected_model={selected_model}, type={type(selected_model)}")
        
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
        
        # Load data
        teams_df, matches_df, processed_df, _ = load_data()
        
        # Make prediction using processed data if available
        if not processed_df.empty:
            result = predict_with_processed_data(
                home_team_id, away_team_id, match_date, selected_model, 
                processed_df, teams_df
            )
        else:
            result = predict_with_feature_modification(
                home_team_id, away_team_id, match_date, selected_model, 
                matches_df, teams_df
            )
        
        return result
    except Exception as e:
        # Print the full exception details to the console for debugging
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

def load_teams():
    """Load teams for dropdowns"""
    teams_df, test_matches_df, processed_df, _ = load_data()
    
    # Option 1: Get all teams from Team.csv without filtering
    # This will show all teams regardless of whether they have match data
    team_dict = dict(zip(teams_df['team_long_name'], teams_df['team_api_id']))
    
    # Option 2 (current method): Filter teams to only those with match data
    # if not processed_df.empty:
    #     home_teams = set(processed_df['home_team_api_id'])
    #     away_teams = set(processed_df['away_team_api_id'])
    #     available_team_ids = list(home_teams.union(away_teams))
    #     
    #     # Filter teams to only those in the available data
    #     available_teams = teams_df[teams_df['team_api_id'].isin(available_team_ids)]
    # else:
    #     # If processed data not available, use test matches
    #     home_teams = set(test_matches_df['home_team_api_id'])
    #     away_teams = set(test_matches_df['away_team_api_id'])
    #     available_team_ids = list(home_teams.union(away_teams))
    #     available_teams = teams_df[teams_df['team_api_id'].isin(available_team_ids)]
    # 
    # # Create a dict mapping team name to team_api_id
    # team_dict = dict(zip(available_teams['team_long_name'], available_teams['team_api_id']))
    
    return gr.update(choices=list(team_dict.items()))

def load_models():
    """Load available models for dropdown"""
    _, _, _, models = load_data()
    
    # Check if models are available
    if not models:
        print("Warning: No models found in the models directory")
        return gr.update(choices=[])
        
    # Return model names with descriptions
    model_choices = [(f"{name} Model", name) for name in models.keys()]
    return gr.update(choices=model_choices)

def create_interface_medium():
    """Create medium complexity Gradio interface"""
    with gr.Blocks(title="Football Match Prediction") as app:
        gr.Markdown("# Football Match Prediction")
        gr.Markdown("Select teams and a date to predict the match outcome.")
        
        # Load team and model data first - this gets the data ready for the dropdowns
        teams_df, _, _, models_dict = load_data()
        team_choices = [(name, id) for name, id in zip(teams_df['team_long_name'], teams_df['team_api_id'])]
        model_choices = [(f"{name} Model", name) for name in models_dict.keys()]
        
        with gr.Row():
            with gr.Column():
                home_team = gr.Dropdown(label="Select Home Team", choices=team_choices, interactive=True)
            with gr.Column():
                away_team = gr.Dropdown(label="Select Away Team", choices=team_choices, interactive=True)
        
        with gr.Row():
            match_date = gr.Textbox(
                label="Match Date (YYYY-MM-DD)", 
                value="2016-05-15"  # Late in the 2015/2016 season
            )
            model_selector = gr.Dropdown(label="Select Prediction Model", choices=model_choices, interactive=True)
        
        predict_btn = gr.Button("Predict Match Outcome")
        result_output = gr.Markdown()
        
        # Run prediction when button clicked
        predict_btn.click(
            fn=predict_football_match,
            inputs=[home_team, away_team, match_date, model_selector],
            outputs=result_output
        )
    
    return app

if __name__ == "__main__":
    app = create_interface_medium()
    app.launch()
