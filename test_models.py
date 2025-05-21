import os
import joblib
import json
import pandas as pd
import numpy as np

def test_model_loading():
    """Test loading of all models to verify they work correctly"""
    models_dir = 'models'
    data_dir = 'data'
    
    print("Testing model loading...")
    
    # Get all model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    model_names = [f.replace('_best.pkl', '') for f in model_files if '_best.pkl' in f]
    
    if not model_names:
        print("No models found in models/ directory")
        return False
    
    print(f"Found {len(model_names)} models: {', '.join(model_names)}")        # Try to load X_test to test prediction
    try:
        X_test = pd.read_csv(os.path.join(data_dir, 'feature', 'X_test.csv'))
        print(f"Loaded test data with {X_test.shape[0]} samples and {X_test.shape[1]} features")
        
        if X_test.shape[0] == 0:
            print("Warning: Test data is empty")
            return False
            
        test_sample = X_test.iloc[[0]]
        print(f"Using first sample (home_team: {test_sample['home_team_api_id'].iloc[0]}, away_team: {test_sample['away_team_api_id'].iloc[0]})")
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        return False
    
    # Try loading each model
    success = True
    for model_name in model_names:
        try:
            print(f"\nTesting model: {model_name}")
            
            # Load model
            model_path = os.path.join(models_dir, f"{model_name}_best.pkl")
            metadata_path = os.path.join(models_dir, f"{model_name}_best_metadata.json")
            
            model = joblib.load(model_path)
            print(f"✓ Model loaded successfully")
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"✓ Metadata loaded, model has {len(metadata['features'])} features")
            
            # Check required features are present
            required_features = metadata['features']
            missing_features = [f for f in required_features if f not in test_sample.columns]
            if missing_features:
                print(f"✗ Warning: {len(missing_features)} features are missing from test data")
                print(f"  First few missing: {missing_features[:5]}")
                # Add dummy columns
                for feature in missing_features:
                    test_sample[feature] = np.nan
            else:
                print(f"✓ All required features are present in test data")
            
            # Select only required features
            X = test_sample[required_features]
            
            # Try prediction
            y_proba = model.predict_proba(X)
            print(f"✓ Prediction successful!")
            print(f"  Result: Loss: {y_proba[0][0]:.2f}, Draw: {y_proba[0][1]:.2f}, Win: {y_proba[0][2]:.2f}")
            
        except Exception as e:
            print(f"✗ Error testing {model_name}: {str(e)}")
            success = False
    
    return success

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n✅ All tests completed successfully, models should work with the app")
    else:
        print("\n❌ Some tests failed, please check the errors above")
