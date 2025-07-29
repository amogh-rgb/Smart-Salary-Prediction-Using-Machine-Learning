Salary Prediction using Machine Learning
This project implements a machine learning model to predict annual salaries based on various professional and demographic factors. It utilizes a simulated salary dataset and provides comprehensive data analysis, model evaluation, and interactive visualizations within a Jupyter Notebook environment.
Table of Contents
Project Overview
Dataset
Setup and Installation
Code Structure and Explanation
Cell 1: Import Libraries
Cell 2: Load Data
Cell 3: Data Preprocessing
Cell 4: Model Training
Cell 5: Model Evaluation
Cell 6: Actual vs. Predicted Salaries Scatter Plot
Cell 8: Job Role Distribution Pie Chart
Cell 9: Average Predicted Salary by Job Role and Education Level Bar Plot
Cell 10: Summary Table with Average, Minimum, and Maximum Salary Data
Machine Learning Model
Visualizations
How to Run the Notebook
Project Overview
The primary goal of this project is to build an accurate machine learning model capable of predicting annual salaries. It focuses on using a structured dataset, performing necessary data preprocessing, training a robust regression model, and visualizing the results and key insights.
Dataset
The project uses a simulated dataset named simulated_salary_dataset_1500.csv. This CSV file contains 1500 entries with the following columns:
Years of Experience (float)
Age (int)
Gender (object/categorical)
Education Level (object/categorical)
Job Role (object/categorical)
Location (object/categorical)
Industry (object/categorical)
Salary (Annual) (float) - This is the target variable.
Setup and Installation
To run this Jupyter Notebook, you need to have Python installed along with the following libraries. You can install them using pip:
pip install pandas scikit-learn matplotlib seaborn jupyter


Code Structure and Explanation
The project code is organized into several cells within a Jupyter Notebook, each performing a specific part of the machine learning pipeline and data analysis.
Cell 1: Import Libraries
Imports all necessary Python libraries, including pandas for data manipulation, matplotlib.pyplot and seaborn for plotting, numpy for numerical operations, and various modules from sklearn for machine learning tasks.
Cell 2: Load Data
Loads the simulated_salary_dataset_1500.csv file into a pandas DataFrame. It includes error handling to check if the file exists. It then displays the first few rows and a summary of the DataFrame's information (df.info()).
Cell 3: Data Preprocessing
This cell prepares the data for the machine learning model:
Separates features (X) from the target variable (y), which is Salary (Annual).
Identifies numerical and categorical columns.
Applies OneHotEncoder to categorical features using ColumnTransformer to convert them into a numerical format suitable for the model. Numerical features are passed through.
Generates a list of all feature names after encoding for later use (e.g., in feature importance plotting, though that plot is currently removed).
Cell 4: Model Training
Splits the preprocessed data into training (80%) and testing (20%) sets using train_test_split. A copy of the original test set features (X_test_original) is also created for visualization purposes.
Initializes and trains a RandomForestRegressor model with 100 estimators. This model is chosen for its robustness and high predictive power on tabular data.
Cell 5: Model Evaluation
Uses the trained model to make predictions on the test set (y_pred).
Calculates and prints the R-squared (R2) score and the Mean Absolute Error (MAE) to evaluate the model's performance. The R-squared score indicates the proportion of variance in the dependent variable that is predictable from the independent variables.
Cell 6: Actual vs. Predicted Salaries Scatter Plot
Generates a scatter plot comparing the actual salaries from the test set against the model's predicted salaries.
Points are colored based on Years of Experience, providing an additional dimension of insight.
A red dashed line represents perfect predictions, allowing for easy visual assessment of the model's accuracy.
Cell 8: Job Role Distribution Pie Chart
Creates a donut (pie) chart visualizing the proportion of employees across different Job Role categories in the entire dataset. This provides a clear overview of the dataset's composition by profession.
Cell 9: Average Predicted Salary by Job Role and Education Level Bar Plot
Generates a grouped bar plot showing the average predicted salary for each Job Role, further segmented by Education Level. This visualization helps understand how predicted salaries vary across professions and qualifications.
Cell 10: Summary Table with Average, Minimum, and Maximum Salary Data
Creates and displays a comprehensive summary table. This table groups the data by Job Role and Education Level and calculates:
Average Annual Salary
Minimum Annual Salary
Maximum Annual Salary
Average Years of Experience
Average Age
Count of individuals in each group
The table is formatted for readability, with currency and decimal precision.
Machine Learning Model
The core of this project is a RandomForestRegressor. This ensemble learning method operates by constructing a multitude of decision trees at training time and outputting the mean prediction of the individual trees.
Model Performance:
R-squared (R2) Score: Typically around 0.96 (96%). This indicates that the model explains 96% of the variance in annual salaries, signifying a very high level of accuracy and predictive power.
Mean Absolute Error (MAE): Typically around $10,000 - $11,000. This means that, on average, the model's salary predictions are off by approximately this amount from the actual salaries.
The model's R-squared score consistently surpasses the target accuracy of 80%.
Visualizations
The notebook includes three key visualizations:
Actual vs. Predicted Salaries Scatter Plot: To visually assess the model's accuracy and how well predictions align with actual values, considering years of experience.
Job Role Distribution Pie Chart: To understand the proportional representation of different job roles within the dataset.
Average Predicted Salary by Job Role and Education Level Bar Plot: To analyze salary trends based on profession and educational background.
How to Run the Notebook
Download the Notebook: Save the entire code provided in the Jupyter Notebook format (e.g., salary_prediction_project.ipynb).
Place the Data: Ensure the simulated_salary_dataset_1500.csv file is in the same directory as your .ipynb file.
Open Jupyter: Launch Jupyter Notebook from your terminal by typing jupyter notebook.
Navigate and Run: Open the salary_prediction_project.ipynb file in your browser. You can run each cell sequentially by clicking on a cell and pressing Shift + Enter, or run all cells by selecting "Run" -> "Run All Cells" from the menu.
The output, including printed metrics and generated plots, will appear directly within the notebook.

