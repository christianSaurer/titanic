#BELOW SCRIPT SHOWS SOME EXAMPLES HOW TO USE THE MAIN.PY.
#IF YOU RUN MAIN.PY DIRECTLY, IT WILL BY DEFAULT USE THE BEST HYPERPARAMETERS THAT I'VE MANAGED TO VALIDATE
#THROUGH KAGGLE SUBMISSION PROCESS.

from main import *

#best random forest configuration found, accuracy: 0.78708.
#running main script directly will default to those parameters.
model_type = 'random_forest'
hyper_parameters = {'n_estimators': 500, 'max_samples': 100, 'bootstrap': True, 'n_jobs': -1, 'oob_score': True}
filename = '_'.join(['rf'] + [str(value) for value in hyper_parameters.values()])
main(model_type, filename, hyper_parameters=hyper_parameters)


#grid search on different random forest hyperparamsters. saves the best one found (cross validation, 5 fold)
model_type = 'random_forest_grid_search'
param_grid = [{'n_estimators': [10, 100, 200, 500, 600, 800, 1000]
               ,'min_samples_leaf':[1,2,5,10,20,50]
                ,'max_features': ['auto', 'sqrt', 'log2']
               ,'max_samples': [20,100,200]
               ,'n_jobs': [-1]}]
filename = 'rf_grid_search'
main(model_type, filename, grid=param_grid)


#trying out a bunch of hyperparameters that achieved high cv score
model_type = 'random_forest'
for hyper_params in [{'max_features': 'auto', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 500, 'n_jobs': -1,'bootstrap': True, 'oob_score': True}
                        ,{'max_features': 'auto', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 600, 'n_jobs': -1,'bootstrap': True, 'oob_score': True}
                        , {'max_features': 'auto', 'max_samples': 200, 'min_samples_leaf': 2, 'n_estimators': 10, 'n_jobs': -1,'bootstrap': True, 'oob_score': True}
                        , {'max_features': 'sqrt', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 200, 'n_jobs': -1,'bootstrap': True, 'oob_score': True}
                        , {'max_features': 'sqrt', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 500, 'n_jobs': -1,'bootstrap': True, 'oob_score': True}
                        , {'max_features': 'sqrt', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 800, 'n_jobs': -1,'bootstrap': True, 'oob_score': True}
                        , {'max_features': 'log2', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 200, 'n_jobs': -1,'bootstrap': True, 'oob_score': True}
                        , {'max_features': 'log2', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 1000, 'n_jobs': -1,'bootstrap': True, 'oob_score': True}
]:
    filename = '_'.join(['rf'] + [str(value) for value in hyper_params.values()])
    main(model_type, filename, hyper_parameters=hyper_params)


#see below output of cv scores obtained by grid search above
#0.8316301550436258 {'max_features': 'auto', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 500, 'n_jobs': -1}
# 0.8338773460548616 {'max_features': 'auto', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 600, 'n_jobs': -1}
# 0.8305128366078713 {'max_features': 'auto', 'max_samples': 200, 'min_samples_leaf': 2, 'n_estimators': 10, 'n_jobs': -1}
# 0.8350134957002071 {'max_features': 'sqrt', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 200, 'n_jobs': -1}
# 0.8327600276191074 {'max_features': 'sqrt', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 500, 'n_jobs': -1}
# 0.8305191136777352 {'max_features': 'sqrt', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 800, 'n_jobs': -1}
# 0.8316427091833531 {'max_features': 'log2', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 200, 'n_jobs': -1}
# 0.8338773460548616 {'max_features': 'log2', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 1000, 'n_jobs': -1}
# 0.8271608813006088 {'max_features': 'auto', 'max_samples': 100, 'min_samples_leaf': 1, 'n_estimators': 100, 'n_jobs': -1}
# 0.8282907538760907 {'max_features': 'auto', 'max_samples': 100, 'min_samples_leaf': 1, 'n_estimators': 200, 'n_jobs': -1}
# 0.8226727763480008 {'max_features': 'auto', 'max_samples': 100, 'min_samples_leaf': 1, 'n_estimators': 500, 'n_jobs': -1}
# 0.8260372857949909 {'max_features': 'auto', 'max_samples': 100, 'min_samples_leaf': 1, 'n_estimators': 600, 'n_jobs': -1}
# 0.8271608813006089 {'max_features': 'auto', 'max_samples': 100, 'min_samples_leaf': 1, 'n_estimators': 800, 'n_jobs': -1}
# 0.8226727763480006 {'max_features': 'auto', 'max_samples': 100, 'min_samples_leaf': 1, 'n_estimators': 1000, 'n_jobs': -1}
# 0.827154604230745 {'max_features': 'auto', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 100, 'n_jobs': -1}
# 0.8271420500910175 {'max_features': 'auto', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 200, 'n_jobs': -1}
# 0.8271483271608814 {'max_features': 'auto', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 800, 'n_jobs': -1}
# 0.8282593685267716 {'max_features': 'auto', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 1000, 'n_jobs': -1}
# 0.821542903772519 {'max_features': 'auto', 'max_samples': 200, 'min_samples_leaf': 2, 'n_estimators': 100, 'n_jobs': -1}
# 0.8226602222082733 {'max_features': 'auto', 'max_samples': 200, 'min_samples_leaf': 2, 'n_estimators': 200, 'n_jobs': -1}
# 0.8215429037725188 {'max_features': 'auto', 'max_samples': 200, 'min_samples_leaf': 2, 'n_estimators': 500, 'n_jobs': -1}
# 0.8226853304877284 {'max_features': 'sqrt', 'max_samples': 100, 'min_samples_leaf': 1, 'n_estimators': 100, 'n_jobs': -1}
# 0.8226790534178645 {'max_features': 'sqrt', 'max_samples': 100, 'min_samples_leaf': 1, 'n_estimators': 200, 'n_jobs': -1}
# 0.8215617349821104 {'max_features': 'sqrt', 'max_samples': 100, 'min_samples_leaf': 1, 'n_estimators': 500, 'n_jobs': -1}
# 0.8282844768062269 {'max_features': 'sqrt', 'max_samples': 100, 'min_samples_leaf': 1, 'n_estimators': 600, 'n_jobs': -1}
# 0.8226853304877284 {'max_features': 'sqrt', 'max_samples': 100, 'min_samples_leaf': 1, 'n_estimators': 800, 'n_jobs': -1}
# 0.8260372857949909 {'max_features': 'sqrt', 'max_samples': 100, 'min_samples_leaf': 1, 'n_estimators': 1000, 'n_jobs': -1}
# 0.8248948590797814 {'max_features': 'sqrt', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 10, 'n_jobs': -1}
# 0.828278199736363 {'max_features': 'sqrt', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 100, 'n_jobs': -1}
# 0.8260310087251271 {'max_features': 'sqrt', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 600, 'n_jobs': -1}
# 0.8271420500910176 {'max_features': 'sqrt', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 1000, 'n_jobs': -1}
# 0.822666499278137 {'max_features': 'log2', 'max_samples': 100, 'min_samples_leaf': 1, 'n_estimators': 200, 'n_jobs': -1}
# 0.8294017952419811 {'max_features': 'log2', 'max_samples': 100, 'min_samples_leaf': 1, 'n_estimators': 500, 'n_jobs': -1}
# 0.8282844768062269 {'max_features': 'log2', 'max_samples': 100, 'min_samples_leaf': 1, 'n_estimators': 600, 'n_jobs': -1}
# 0.821542903772519 {'max_features': 'log2', 'max_samples': 100, 'min_samples_leaf': 1, 'n_estimators': 800, 'n_jobs': -1}
# 0.8249199673592367 {'max_features': 'log2', 'max_samples': 100, 'min_samples_leaf': 1, 'n_estimators': 1000, 'n_jobs': -1}
# 0.8249325214989645 {'max_features': 'log2', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 10, 'n_jobs': -1}
# 0.8294143493817085 {'max_features': 'log2', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 100, 'n_jobs': -1}
# 0.8293892411022536 {'max_features': 'log2', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 500, 'n_jobs': -1}
# 0.8282593685267716 {'max_features': 'log2', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 600, 'n_jobs': -1}
# 0.8293955181721172 {'max_features': 'log2', 'max_samples': 200, 'min_samples_leaf': 1, 'n_estimators': 800, 'n_jobs': -1}
# 0.8058313979034587 {'max_features': 'log2', 'max_samples': 200, 'min_samples_leaf': 2, 'n_estimators': 10, 'n_jobs': -1}
