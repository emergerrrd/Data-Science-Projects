'''
This python package aims to test the functions
in churn_library.py.

Author: Emerson Chua
Date: 10/12/2021
'''

import os
import logging
import churn_library as cl

logging.basicConfig(
    filename=f"./logs/churn_library_{time.strftime('%b_%d_%Y_%H_%M_%S')}.log",
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("Testing import_data: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return df


def test_eda(perform_eda, df):
    '''
    test perform eda function
    '''
    try:
        perform_eda(df)
    except AttributeError as err:
        logging.error("Testing perform_eda: Column(s) missing from dataframe")
        raise err

    path = "./images/eda"

    try:
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.warning(
            "Testing perform_eda: EDA images were not save to the eda folder")
        raise err


def test_encoder_helper(encoder_helper, df):
    '''
    test encoder helper
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    df = encoder_helper(df, cat_columns, 'Churn')

    try:
        for col in cat_columns:
            assert col in df.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: Dataframe is missing transformed categorical columns")
        return err

    return df


def test_perform_feature_engineering(perform_feature_engineering, df):
    '''
    test perform_feature_engineering
    '''
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')
    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: Missing output")
        raise err

    return X_train, X_test, y_train, y_test


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    test train_models
    '''
    train_models(X_train, X_test, y_train, y_test)
    path = "./images/results/"
    try:
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
    except FileNotFoundError as err:
        logging.error("Testing train_models: Results image files not found")
        raise err

    path = "./models/"
    try:
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: Model files not found")
        raise err


if __name__ == "__main__":
    BANK_DF = test_import(cl.import_data)
    test_eda(cl.perform_eda, BANK_DF)
    BANK_DF = test_encoder_helper(cl.encoder_helper, BANK_DF)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cl.perform_feature_engineering, BANK_DF)
    test_train_models(cl.train_models, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
