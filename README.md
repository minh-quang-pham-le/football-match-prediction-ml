# ⚽ Predicting Football Match Outcomes using Traditional Machine Learning ⚽

This project aims to predict the outcomes of football matches (Win/Draw/Loss) using traditional machine learning techniques such as Logistic Regression, Decision Trees, Random Forests, and Gradient Boosting. It leverages the European Soccer Database from Kaggle, containing over 25,000 matches between 2008 and 2016. The project includes an interactive Gradio demo and is packaged with Docker for reproducibility.

---

## Getting Started

These instructions will help you set up the project on your local machine for development, testing, and running the Gradio demo.

### Prerequisites

Make sure the following software is installed:

- Python >= 3.9  
- pip  
- Docker (for deployment)  
- git  

### Installing

And you can install all required packages automatically via:

```
python setup.py
```

Or manually

```
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r docker/requirements.txt
```

## How to Run the Project

1. **Clone the repository**

```
git clone https://github.com/yourusername/Football-Match-Prediction-ML.git
cd Football-Match-Prediction-ML
```

2. **Run the Full Pipeline (Train + Evaluate)**

```
python run_pipeline.py
```
This script will:

- Load and clean data (data/raw/)

- Generate features (data/features/)

- Train and save model (models/)

- Evaluate performance on the training set
  

3. **Launch Gradio Demo App**

```
python demo/app.py
```

4. **Run Everything with Docker (Deployment)**

Build the Docker image:
```
docker build -t football-predictor .
```
Run the container (Gradio mode):
```
docker run -p 7860:7860 football-predictor
```
Then go to: http://localhost:7860 to access Gradio app

---

## Running the Tests

To ensure everything works as expected, you can run unit tests and data integrity checks

### 🧪 End-to-End Tests

These test the full pipeline from data loading to prediction:

```
pytest tests/
```

---

## Built With

- [Scikit-learn](https://scikit-learn.org/) – Machine learning models  
- [Pandas](https://pandas.pydata.org/) – Data manipulation  
- [Gradio](https://gradio.app/) – Web interface  
- [Docker](https://www.docker.com/) – Containerization  
- [Matplotlib](https://matplotlib.org/) – Visualization  
See full list of [contributors](https://github.com/minh-quang-pham-le/Football-Match-Prediction-ML/contributors)

---

## Authors

- **Phạm lê Minh Quang** – *Group Leader*  
- **Phạm Ngọc Thạch**  
- **Nguyễn Hoàng Thái** 
- **Đặng Xuân Đạt** 
- **Nguyễn Bảo Lâm** 

See the list of [contributors](https://github.com/minh-quang-pham-le/Football-Match-Prediction-ML/contributors) for all who participated in this project.

---

## License

This project is licensed under the MIT License – see the [LICENSE.md](LICENSE.md) file for details.

---

## Acknowledgments

- Dataset from [Kaggle: European Soccer Database](https://www.kaggle.com/datasets/hugomathien/soccer)  
- Mini Project from HUST – IT3190E Machine Learning Course  
- Inspiration from sports analytics research communities  
