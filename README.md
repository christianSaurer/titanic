# Readme
This project can be used to create Random forest models to build submissions for the well-known titanic project on kaggle (https://www.kaggle.com/c/titanic).

The following functionalities are covered:

- **retrieve_files.py:** connects to kaggle api and downloads all required input data. You will first need to download an API token (see https://www.kaggle.com/docs/api).

- **exploration.py:** basic data exploration, relying on ppscore library, which creates non-linear, non-symmetric correlation metrics.

- **main.py:** trains RF model with default parameters and creates predictions on test set. Accuracy: 0.78947. Features used include mostly the columns contained in the raw input dataset (see https://www.kaggle.com/c/titanic/data for detailed description of the meaning of the column names).
	- Numerical columns: 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'. All missing values are replaced by Median values of training set.
	- Categorical columns: 'Sex', 'Embarked', 'title', 'Cabin', 'has_nickname'. All are one-hot encoded. Missing values are replaced by mode of training set. 'title' is a feature extracted from the 'name' field (e.g. Dr., Mrs., Mr.). Only the first letter of the original 'Cabin' column is used. 'has_nickname' is a boolean (1 if person has indicated a nickname, else 0). 

- **prediction_test_script.py:** demonstrates how to invoke main.py with custom Random forest hyperparameter and how to perform grid search on the RF hyperparameters.

The project relies on python 3.9 and uses the packages pandas, sklearn and nameparser for feature engineering and prediction. Libraries ppscore and seaborn are used for exploration. The library kaggle is used to use the kaggle api to download competition files.
