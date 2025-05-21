# Football Match Prediction

This application predicts the outcome of football matches using pre-trained machine learning models. The app allows users to select teams, players, and match details to generate predictions for match outcomes.

## Features

- Select home and away teams
- Choose 11 players for each team
- Set match date (which affects form calculations)
- Select from multiple prediction models
- View prediction results showing win/draw probabilities

## Setup

### Prerequisites

- Python 3.8 or higher
- All dependencies listed in `docker/requirements.txt`

### Installation

1. Clone the repository:
```
git clone <repository-url>
cd football-match-prediction-ml
```

2. Install dependencies:
```
pip install -r docker/requirements.txt
```

3. Download data (if not already present):
```
python src/data/download_data.py
```

## Usage

There are two application versions:

### Full Version

The full version allows you to select teams, players, date, and model:

```
python app.py
```

Steps:
1. Select a home team
2. Select 11 players for the home team
3. Select an away team
4. Select 11 players for the away team
5. Enter a match date (must be during the 2015/2016 season)
6. Choose a prediction model
7. Click "Predict Match Outcome"

### Simple Version

The simple version allows you to select from existing test matches:

```
python app_simple.py
```

Steps:
1. Select a match from the dropdown (these are actual matches from the test set)
2. Choose a prediction model
3. Click "Predict Match Outcome"

## How It Works

1. User inputs are processed to generate features needed by the models
2. Features include team statistics, player attributes, recent form, and head-to-head history
3. The selected model predicts probabilities for the three possible outcomes: Home Win, Draw, Away Win
4. Results are displayed showing the probability of each outcome

## Models

The application includes several pre-trained models:
- LightGBM
- XGBoost
- RandomForest
- GradientBoost
- LogisticRegression

Each model has been trained on matches from 2008/2009 to 2014/2015, with the 2015/2016 season used for testing.

## Data

The application uses the European Soccer Database from Kaggle, which includes:
- Match results
- Player attributes
- Team attributes
- Betting odds

## Notes

- Predictions are only available for the 2015/2016 season (test set)
- The application ensures that all 82 features required by the models are correctly generated
- The interface is built using Gradio for ease of use
