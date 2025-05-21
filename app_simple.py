import os
import pandas as pd
import numpy as np
import gradio as gr
import joblib
import json
from typing import Dict, List

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
    
    # Create a dict of available matches for the dropdown
    team_names = dict(zip(teams_df['team_api_id'], teams_df['team_long_name']))
    available_matches = []
    
    # Use matches from processed data if available, otherwise use raw data
    if not processed_df.empty:
        match_source = processed_df
        print("Using processed data for match list")
    else:
        match_source = test_matches_df
        print("Using raw data for match list")
    
    for i, match in match_source.iterrows():
        if i >= 100:  # Limit to first 100 matches to avoid huge dropdown
            break
            
        if 'home_team_api_id' in match and 'away_team_api_id' in match:
            home_id = match['home_team_api_id']
            away_id = match['away_team_api_id']
            
            home_name = team_names.get(home_id, f"Team {home_id}")
            away_name = team_names.get(away_id, f"Team {away_id}")
            
            # Use match date if available, otherwise use index
            if 'date' in match:
                date_str = f"({match['date']})"
            else:
                date_str = f"(Match #{i})"
                
            # Use match_api_id if available, otherwise use row index
            if 'match_api_id' in match:
                match_id = match['match_api_id']
            else:
                match_id = i
                
            match_key = f"{match_id}: {home_name} vs {away_name} {date_str}"
            available_matches.append((match_key, match_id))
    
    # Get all available models
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    models = {f.replace('_best.pkl', ''): f for f in model_files if '_best.pkl' in f}
    
    return teams_df, test_matches_df, processed_df, available_matches, models

def load_model(model_name):
    """Load model and metadata by name"""
    model_path = os.path.join(MODELS_DIR, f"{model_name}_best.pkl")
    metadata_path = os.path.join(MODELS_DIR, f"{model_name}_best_metadata.json")
    
    model = joblib.load(model_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return model, metadata

def prepare_match_features(match_id, matches_df, processed_df, teams_df):
    """Extract features for a single match"""
    # First try to find the match in processed data if available
    if not processed_df.empty:
        # Check if match_id is in processed_df
        if 'match_api_id' in processed_df.columns:
            match_features = processed_df[processed_df['match_api_id'] == match_id]
            if not match_features.empty:
                match = matches_df[matches_df['match_api_id'] == match_id].iloc[0] if match_id in matches_df['match_api_id'].values else None
                
                # Extract team IDs either from match or from match_features
                if match is not None:
                    home_team_id = match['home_team_api_id']
                    away_team_id = match['away_team_api_id']
                else:
                    home_team_id = match_features['home_team_api_id'].iloc[0]
                    away_team_id = match_features['away_team_api_id'].iloc[0]
                
                # Get team names
                home_team_name = teams_df[teams_df['team_api_id'] == home_team_id]['team_long_name'].iloc[0] if home_team_id in teams_df['team_api_id'].values else f"Team {home_team_id}"
                away_team_name = teams_df[teams_df['team_api_id'] == away_team_id]['team_long_name'].iloc[0] if away_team_id in teams_df['team_api_id'].values else f"Team {away_team_id}"
                
                return match_features, home_team_name, away_team_name
        # If match_id is just an index in processed_df
        elif match_id < len(processed_df):
            match_features = processed_df.iloc[[match_id]].copy()
            
            # Extract team IDs from match_features
            home_team_id = match_features['home_team_api_id'].iloc[0]
            away_team_id = match_features['away_team_api_id'].iloc[0]
            
            # Get team names
            home_team_name = teams_df[teams_df['team_api_id'] == home_team_id]['team_long_name'].iloc[0] if home_team_id in teams_df['team_api_id'].values else f"Team {home_team_id}"
            away_team_name = teams_df[teams_df['team_api_id'] == away_team_id]['team_long_name'].iloc[0] if away_team_id in teams_df['team_api_id'].values else f"Team {away_team_id}"
            
            return match_features, home_team_name, away_team_name
    
    # If not found in processed data or processed data isn't available, try X_test
    try:
        # Get the match data from matches_df if available
        if match_id in matches_df['match_api_id'].values:
            match = matches_df[matches_df['match_api_id'] == match_id].iloc[0]
            
            # Load test features 
            X_test = pd.read_csv(os.path.join(DATA_DIR, 'feature', 'X_test.csv'))
            
            # Find this match in the features
            match_features = X_test[X_test['match_api_id'] == match_id]
            
            # If match not found in features, return None
            if match_features.empty:
                return None, None, None
            
            # Get team names
            home_team_name = teams_df[teams_df['team_api_id'] == match['home_team_api_id']]['team_long_name'].iloc[0]
            away_team_name = teams_df[teams_df['team_api_id'] == match['away_team_api_id']]['team_long_name'].iloc[0]
            
            return match_features, home_team_name, away_team_name
    except Exception as e:
        print(f"Error finding match in test features: {e}")
    
    # If all else fails
    return None, None, None

def predict_match(match_features, model, metadata):
    """Make prediction with the selected model"""
    # Ensure we have all required features
    required_features = metadata['features']
    
    # Create missing columns with NaN values
    for feature in required_features:
        if feature not in match_features.columns:
            match_features[feature] = np.nan
    
    # Select only the features required by the model
    X = match_features[required_features]
    
    # Predict probabilities
    y_proba = model.predict_proba(X)
    
    # Map back to outcomes
    return {
        'Loss': y_proba[0][0],
        'Draw': y_proba[0][1],
        'Win': y_proba[0][2]
    }

def predict_football_match(match_id, selected_model):
    """Main prediction function for the Gradio interface"""
    try:
        match_id = int(match_id)
        
        # Load data
        teams_df, matches_df, processed_df, _, _ = load_data()
        
        # Get match features
        match_features, home_team_name, away_team_name = prepare_match_features(match_id, matches_df, processed_df, teams_df)
        
        if match_features is None:
            return "Error: Match not found in the feature data."
        
        # Load selected model
        model, metadata = load_model(selected_model)
        
        # Make prediction
        prediction = predict_match(match_features, model, metadata)
        
        # Format the output
        result = f"## Match Prediction: {home_team_name} vs {away_team_name}\n\n"
        result += f"**Win Probability for {home_team_name}:** {prediction['Win']*100:.2f}%\n\n"
        result += f"**Draw Probability:** {prediction['Draw']*100:.2f}%\n\n"
        result += f"**Win Probability for {away_team_name}:** {prediction['Loss']*100:.2f}%\n\n"
        result += f"**Model Used:** {selected_model}\n"
        
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def load_matches():
    """Load available matches for dropdown"""
    _, _, _, available_matches, _ = load_data()
    return gr.Dropdown.update(choices=available_matches)

def load_models():
    """Load available models for dropdown"""
    _, _, _, _, models = load_data()
    # Return model names with descriptions for better display
    model_choices = [(f"{name} Model", name) for name in models.keys()]
    return gr.Dropdown.update(choices=model_choices)

def create_interface_simple():
    """Create simplified Gradio interface for testing"""
    with gr.Blocks(title="Football Match Prediction (Simple Version)") as app:
        gr.Markdown("# Football Match Prediction - Simple Tester")
        gr.Markdown("Select a match and model to predict the outcome.")
        
        with gr.Row():
            match_selector = gr.Dropdown(label="Select Match", interactive=True)
            model_selector = gr.Dropdown(label="Select Prediction Model", interactive=True)
        
        predict_btn = gr.Button("Predict Match Outcome")
        result_output = gr.Markdown()
        
        # Load matches and models when the app starts
        match_selector.update = load_matches
        model_selector.update = load_models
        
        # Run prediction when button clicked
        predict_btn.click(
            fn=predict_football_match,
            inputs=[match_selector, model_selector],
            outputs=result_output
        )
    
    return app

if __name__ == "__main__":
    app = create_interface_simple()
    app.launch()
