@echo off
echo Running notebooks/01_Load_data.ipynb...
jupyter nbconvert --to notebook --execute "notebooks/01_Load_data.ipynb" --output "notebooks/01_Load_data_output.ipynb"
echo Finished notebooks/01_Load_data.ipynb

echo Running notebooks/02_Feature_Engineering.ipynb...
jupyter nbconvert --to notebook --execute "notebooks/02_Feature_Engineering.ipynb" --output "notebooks/02_Feature_Engineering_output.ipynb"
echo Finished notebooks/02_Feature_Engineering.ipynb

echo Running notebooks/03_Split_data.ipynb...
jupyter nbconvert --to notebook --execute "notebooks/03_Split_data.ipynb" --output "notebooks/03_Split_data_output.ipynb"
echo Finished notebooks/03_Split_data.ipynb

echo Running notebooks/04_EDA_On_Train.ipynb...
jupyter nbconvert --to notebook --execute "notebooks/04_EDA_On_Train.ipynb" --output "notebooks/04_EDA_On_Train_output.ipynb"
echo Finished notebooks/04_EDA_On_Train.ipynb

echo Running notebooks/05_Model_Training.ipynb...
jupyter nbconvert --to notebook --execute "notebooks/05_Model_Training.ipynb" --output "notebooks/05_Model_Training_output.ipynb"
echo Finished notebooks/05_Model_Training.ipynb

echo Running notebooks/06_Evaluation.ipynb...
jupyter nbconvert --to notebook --execute "notebooks/06_Evaluation.ipynb" --output "notebooks/06_Evaluation_output.ipynb"
echo Finished notebooks/06_Evaluation.ipynb