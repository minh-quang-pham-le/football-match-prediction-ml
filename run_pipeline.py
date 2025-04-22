import argparse
import logging
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import build_features
from src.model_training import train_model
from src.evaluation import evaluate_model

def main(args):
    pass
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the full ML pipeline")
    parser.add_argument('--raw_data', type=str, default='data/raw/matches.csv',
                        help='Path to raw data CSV file')
    args = parser.parse_args()
    # main(args)