This project can be used to create Random forest models to build submissions for the well-known titanic project on kaggle (https://www.kaggle.com/c/titanic).

The following functionalities are covered:

- **retrieve_files.py:** connects to kaggle api and downloads all required input data. You will first need to download an API token (see https://www.kaggle.com/docs/api).

- **exploration.py:** basic data exploration, relying on ppscore library, which creates non-linear, non-symmetric correlation metrics.

- **main.py:** trains RF model with default parameters and creates predictions on test set. Accuracy: 0.78708. Features used include mostly the columns contained in the raw input dataset (see https://www.kaggle.com/c/titanic/data for detailed description of the meaning of the column names).
Numerical columns: 'Pclass', 'Age', 'SibSp', 'Parch', 'Far'. All missing values are replaced by Median values of training set.
categorical columns: 'Sex', 'Embarked', 'title', 'Cabin'. All are one-hot encoded.
	-'title' is a feature extracted from the 'name' field (e.g. Dr., Mrs., Mr.)
	- only the first letter of the original 'Cabin' column is used.

 1 has_nickname (boolean, 1 if person has indicated a nickname, else 0), 2. title: title contained in the name field (e.g. Dr., Mrs., Mr. etc.). The column 'cabin' is modified by extracting only the first letter. Categorical variables are one-hot encoded ('Sex', 'Embarked', 'title', 'Cabin'). Numerical variables 

- **prediction_test_script.py:** demonstrates how to invoke main.py with custom Random forst hyperparameter and how to perform grid search on the RF hyperparameters.
Using the main.py you can build Random Forst models (sklearn) to predict survival of titanic pasengers.

The project relies on python 3.9 and uses the packages pandas, sklearn and nameparser for feature engineering and prediction. Libraries ppscore and seaborn are used for exploration. The library kaggle is used to use the kaggle api to download competition files.
