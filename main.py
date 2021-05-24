import pandas as pd
import os
from nameparser import HumanName
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer


def main(model_type, filename, grid=None, hyper_parameters=None):
    #create dataset containing training and test set
    data_folder = os.getcwd() + '/data/'
    input_data_folder = data_folder + 'input/'
    output_data_folder = data_folder + 'output/'
    if not os.path.exists(output_data_folder):
        os.makedirs(output_data_folder)

    train_df = pd.read_csv(input_data_folder + 'train.csv')
    test_df = pd.read_csv(input_data_folder + 'test.csv')
    cat_variables = ['Sex', 'Embarked', 'title', 'Cabin']
    num_variables = ['Pclass','Age','SibSp','Parch','Fare']
    target_variable = 'Survived'
    index_variable = 'PassengerId'
    test_df[target_variable] = 0
    training_bool = train_df.__len__() * [True] + test_df.__len__() * [False]
    test_bool = [not element for element in training_bool]
    df = pd.concat([train_df, test_df], axis=0)
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)

    #preprocessing: add features, replace missing values, encode categorical variables
    df = feature_engineering(df)
    df_num_processed = preprocessing_numerical(df, num_variables, training_bool)
    df_cat_processed = onehotencoding(df, cat_variables)
    df_processed = pd.concat([df_num_processed, df_cat_processed, df[target_variable] ,df[index_variable]], axis=1)
    df_processed.set_index(index_variable, inplace=True)
    df_train_processed = df_processed.loc[training_bool,:]
    df_test_processed = df_processed.loc[test_bool,:]

    if model_type == 'random_forest_grid_search':
        df_test_processed = cv(df_train_processed,df_test_processed, target_variable, grid)

    elif model_type == 'random_forest':
        df_test_processed = rf(df_train_processed, df_test_processed, target_variable, hyper_parameters)

    df_output = df_test_processed['Survived'].reset_index()
    df_output.to_csv(output_data_folder + filename + '.csv', index=False)


def cv(train, test, target_variable, grid):
    rf = RandomForestClassifier()
    gridsearch = GridSearchCV(rf, grid, cv=5, scoring='accuracy', return_train_score=True, refit=True)
    gridsearch.fit(train.drop(target_variable, axis=1), train[target_variable])
    test[target_variable] = gridsearch.predict(test.drop(target_variable, axis=1))
    scores=gridsearch.cv_results_
    for score,params in zip(scores['mean_test_score'],scores['params']):
        print(score, params)
    gridsearch.best_params_
    return(test)


def rf(train, test, target_variable, hyperparams):
    rf = RandomForestClassifier(**hyperparams)
    rf.fit(train.drop(target_variable,axis=1), train[target_variable])
    test[target_variable] = rf.predict(test.drop(target_variable,axis=1))
    if rf.oob_score:
        print('oob_score:', rf.oob_score_)
    return(test)


def feature_engineering(df):
    df["title"] = df["Name"].apply(lambda x: HumanName(x).title)
    df["first"] = df["Name"].apply(lambda x: HumanName(x).first)
    df["nickname"] = df["Name"].apply(lambda x: HumanName(x).nickname)
    df.loc[df["title"] == '','title'] = df.loc[df["title"] == '', 'first']
    df['has_nickname'] = 1
    df.loc[df["nickname"] == '','has_nickname'] = 0
    df.Cabin = df.Cabin.str[0]
    #df = df[['Survived','PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp',
     #  'Parch', 'Fare', 'Cabin', 'Embarked', 'title', 'has_nickname','train']]
    df.Cabin.fillna('Z', inplace=True)
    df.Embarked.fillna(df.Embarked.mode()[0], inplace=True)
    return(df)


def onehotencoding(df, cat_variables):
    encoder = OneHotEncoder()
    encoder.fit(df[cat_variables])
    transformed_data = encoder.transform(df[cat_variables]).toarray()
    feature_names = encoder.get_feature_names(cat_variables)
    df_transformed = pd.DataFrame(transformed_data, columns=feature_names).astype(int)
    return(df_transformed.reset_index(drop=True))


def preprocessing_numerical(df, num_variables, training_bool):
    df_num = df[num_variables]
    median_imputer = SimpleImputer(strategy='median')
    median_imputer.fit(df_num.loc[training_bool,:])
    df_num_processed = pd.DataFrame(median_imputer.transform(df_num)
                                    ,columns = df_num.columns
                                    ,index=df_num.index)
    return(df_num_processed.reset_index(drop=True))


if __name__ == "__main__":
    #configurations
    model_type = 'random_forest'
    default_hyper_parameters = {'n_estimators': 500, 'max_samples': 100, 'bootstrap': True, 'n_jobs': -1, 'oob_score':True}
    filename = 'default_config_prediction'

    #run model
    main(model_type, filename, hyper_parameters=default_hyper_parameters)
