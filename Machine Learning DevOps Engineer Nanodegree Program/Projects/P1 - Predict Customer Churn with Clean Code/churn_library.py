# library doc string
'''
This python package aims to identify credit card
customers who are most likely to churn.

Author: Emerson Chua
Date: 10/12/2021
'''

# import libraries
import os
from scikitplot.metrics import plot_roc, plot_roc_curve
# from sklearn.metrics import plot_roc_curve,
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import normalize
# import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    print("Importing data from", pth)
    df = pd.read_csv(pth)
    print("Successfully imported data from", pth)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
    output:
            None
    '''
    print("Performing EDA and saving images")

    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Churn Distribution
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig(
        './images/eda/churn_distribution.png')
    plt.close()

    # Customer Age Distribution
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig(
        './images/eda/customer_age_distribution.png')
    plt.close()

    # Marital Status Distribution
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(
        './images/eda/marital_status_distribution.png')
    plt.close()

    # Total Transaction Distribution
    plt.figure(figsize=(20, 10))
    sns.distplot(df['Total_Trans_Ct'])
    plt.savefig(
        './images/eda/total_transaction_distribution.png')
    plt.close()

    # Heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/heatmap.png')
    plt.close()

    print("Successfully performed EDA and saved images")


def encoder_helper(df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new
    column with propotion of churn for each category
    - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that
            could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    print("Encoding categorical columns")

    for col in category_lst:
        new_col = col + '_' + response
        df[new_col] = df.groupby(col)[response].transform("mean")

    print("Successfully encoded categorical columns")
    return df


def perform_feature_engineering(df, response="Churn"):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that
              could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    print("Performing feature engineering")

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                 'Income_Category_Churn', 'Card_Category_Churn']

    y = df[response]
    df = encoder_helper(df, cat_columns, response)
    X = df[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    print("Successfully performed feature engineering")
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    print("Generating classification report images")

    # Random Forest
    plt.figure()
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.figure()
    plt.axis('off')
    plt.savefig(
        './images/results/rf_results.png',
        dpi=300,
        bbox_inches='tight')
    plt.close()

    # Logistic Regression
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(
        './images/results/logistic_results.png',
        dpi=300,
        bbox_inches='tight')
    plt.close()

    print("Successfully generated classification report images")


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    print("Generating feature importance plot")

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth, dpi=300, bbox_inches='tight')
    plt.close()

    print("Successfully generated feature importance plot")


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    print("Training models")

    # Train models
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=400)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    print("Successfully trained models")
    print("Making predictions")

    # Make predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    print("Successfully made predictions")
    print("Saving results as images")

    # Save roc curve
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    lrc_plot = plot_roc(lrc, X_test, y_test, ax=ax)
    rfc_disp = plot_roc(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax)
#     lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('./images/results/roc_curve_result.png')
    plt.close()

    # Save results
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    # Save feature importance
    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_train,
        './images/results/feature_importances.png')

    print("Successfully saved results as images")
    print("Saving models as pickle files")

    # Save pickle files
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    print("Successfully saved models as pickle files")


if __name__ == '__main__':
    BANK_DF = import_data(r"./data/bank_data.csv")
    perform_eda(BANK_DF)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        BANK_DF, 'Churn')
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
