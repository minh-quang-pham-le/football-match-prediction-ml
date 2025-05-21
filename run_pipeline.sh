#!/bin/bash

# Danh sách các notebook với đường dẫn đầy đủ
notebooks=(
    "../notebooks/01_Load_data.ipynb"
    "../notebooks/02_Feature_Engineering.ipynb"
    "../notebooks/03_Split_data.ipynb"
    "../notebooks/04_EDA_On_Train.ipynb"
    "../notebooks/05_Model_Training.ipynb"
    "../notebooks/06_Evaluation.ipynb"
)

# Chạy lần lượt từng notebook
for notebook in "${notebooks[@]}"; do
    echo "Running $notebook..."
    jupyter nbconvert --to notebook --execute "$notebook" --output "$notebook"_output
    echo "Finished $notebook"
done