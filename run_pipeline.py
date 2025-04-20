import argparse
import logging
from src import config
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import build_features
from src.model_training import train_model
from src.evaluation import evaluate_model

def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Step 1: Loading and cleaning data...")
    df_clean = load_and_clean_data(args.raw_data)
    df_clean.to_csv(config.PROCESSED_DATA_PATH + "matches.csv", index=False)

    logging.info("Step 2: Building features...")
    df_feat = build_features(config.PROCESSED_DATA_PATH + "matches.csv")
    df_feat.to_csv(config.FEATURES_PATH + "features.csv", index=False)

    logging.info("Step 3: Training model...")
    X = df_feat.drop(columns=["result"])
    y = df_feat["result"]
    model = train_model(X, y)

    logging.info("Step 4: Evaluating model...")
    evaluate_model(config.MODEL_PATH, X, y)

    logging.info("Pipeline completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the full ML pipeline")
    parser.add_argument('--raw_data', type=str, default='data/raw/matches.csv',
                        help='Path to raw data CSV file')
    args = parser.parse_args()
    # main(args)