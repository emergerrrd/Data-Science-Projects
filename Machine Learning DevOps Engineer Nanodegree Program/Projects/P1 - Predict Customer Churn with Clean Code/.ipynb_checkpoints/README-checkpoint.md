# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The goal of the project was to convert an end-to-end data science notebook into a production-level library that supports the prediction of customer churn.

The project contains the following main files and folders:
* `churn_library.py` is a library with functions that aim to identify credit card customers who are most likely to churn.
* `churn_script_logging_and_test.py` is a library with functions to test the functions of `churn_library.py`.
* `churn_notebook.ipynb` is the notebook `churn_library.py` was built from
* `data` contains the data used in the scripts for analysis and model training
* `images` contains saved images of both EDA and results
* `logs` contains logs from testing
* `models` contains the trained models in the form of .pkl files

Please see requirements.txt for libraries used.

## Running Files
Use the following command to run churn_library.py
```
python churn_library.py
```
Use the following command to test churn_library.py using churn_script_logging_and_tests.py
```
python churn_script_logging_and_tests.py
```

Check for pylint scores using the following commands
```
pylint churn_library.py
pylint churn_script_logging_and_test.py
```

The following commands were used for formatting and improving pylint scores
```
autopep8 churn_library.py --in-place --aggressive
autopep8 churn_script_logging_and_test.py  --in-place --aggressive
```