"""
This churn_library module has multiple functions used to accomplish common tasks in data science

Author: Juan David Millan Cifuentes
Date: Dec 2021
"""
import logging
import os
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)
KEEPS_COLUMNS = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
    "Gender_Churn",
    "Education_Level_Churn",
    "Marital_Status_Churn",
    "Income_Category_Churn",
    "Card_Category_Churn",
]

BASE_PATH = "./images"


class YValues:
    """
    Class to store values for train and test
    """

    def __init__(self, y_train, y_test):
        self.__y_train = y_train
        self.__y_test = y_test

    def y_train(self):
        """
        returns y_train
        """
        return self.__y_train

    def y_test(self):
        """
        returns y_test
        """
        return self.__y_test


def import_data(pth):
    """
    returns dataframe for the csv found at pth
    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    dataframe = pd.DataFrame(pd.read_csv(pth))
    return dataframe


def perform_eda(dataframe):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
    output:
            None
    """

    plt.figure(figsize=(20, 10))
    dataframe["Churn"] = dataframe["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    dataframe["Churn"].hist()
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    plt.title("Churn Rate")
    plt.savefig(os.path.join(f"{BASE_PATH}/eda", "churn_distribution.png"))

    dataframe["Customer_Age"].hist()
    plt.xlabel("Customer Age")
    plt.ylabel("Number of Customers")
    plt.title("Customer Age Distribution")
    plt.savefig(os.path.join(f"{BASE_PATH}/eda", "customer_age_distribution.png"))

    plt.figure(figsize=(20, 10))

    dataframe.Marital_Status.value_counts("normalize").plot(kind="bar")
    plt.xlabel("Marital Status")
    plt.ylabel("Number of Customers")
    plt.title("Customer Marital Status")
    plt.savefig(os.path.join(f"{BASE_PATH}/eda", "marital_status_distribution.png"))

    plt.figure(figsize=(20, 10))

    sns.displot(dataframe["Total_Trans_Ct"]).set(title="Total Transaction Cost")
    plt.savefig(os.path.join(f"{BASE_PATH}/eda", "total_transaction_distribution.png"))

    plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe.corr(), annot=False, cmap="Dark2_r", linewidths=2).set(
        title="Heatmap"
    )
    plt.savefig(os.path.join(f"{BASE_PATH}/eda", "heatmap.png"))


def encode_by(dataframe, category, category_group):
    """
    encode by category_group
    input:
          dataframe: pandas dataframe
          category: Category
          category_group: category_group
    output:
            index Series
    """
    return dataframe[category].apply(lambda row: category_group.loc[row])


def encoder_helper(dataframe, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook
    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that
                      could be used for naming variables or index y column]
    output:
            dataframe: pandas dataframe with new columns for
    """
    try:
        for category in category_lst:
            assert category in dataframe.columns
            assert dataframe[category].shape[0] > 0

            logging.info("Calculating churn proportion for %s column", category)
            category_group = dataframe.groupby(category).mean()[response]
            dataframe[f"{category}_{response}"] = encode_by(
                dataframe, category, category_group
            )
        logging.info("SUCCESS: Encoding of categorical data complete")
        return dataframe
    except AssertionError as err:
        logging.error("ERROR: Encoding of categorical data failed")
        raise err


def perform_feature_engineering(
    dataframe, response
) -> (np.array, np.array, np.array, np.array):
    """
    input:
              dataframe: pandas dataframe
              response: string of response name [optional argument that could
              be used for naming variables or index y column]
    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    """

    x_dataframe = dataframe[KEEPS_COLUMNS]
    y_dataframe = dataframe[response]

    x_train, x_test, y_train, y_test = train_test_split(
        x_dataframe, y_dataframe, test_size=0.3, random_state=42
    )
    return x_train, x_test, y_train, y_test


def classification_report_image(
    response_values: YValues,
    lr_predictions_values: YValues,
    rf_predictions_values: YValues,
):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            response_values: train and test response values
            lr_predictions_values: logistic regression train and test values
            rf_predictions_values: random forest train and test values
    output:
             None
    """
    plt.figure(figsize=(5, 5))
    plt.text(
        0.01, 1.25, str("Random Forest Train"), {"size": 10}, fontproperties="monospace"
    )
    plt.text(
        0.01,
        0.05,
        str(
            classification_report(
                response_values.y_train(), rf_predictions_values.y_train()
            )
        ),
        {"size": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01, 0.6, str("Random Forest Test"), {"size": 10}, fontproperties="monospace"
    )
    plt.text(
        0.01,
        0.7,
        str(
            classification_report(
                response_values.y_test(), rf_predictions_values.y_test()
            )
        ),
        {"size": 10},
        fontproperties="monospace",
    )
    plt.axis("off")
    plt.savefig(f"{BASE_PATH}/results/rf_results.png")

    plt.figure(figsize=(5, 5))
    plt.text(
        0.01,
        1.25,
        str("Linear Regression Train"),
        {"size": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.05,
        str(
            classification_report(
                response_values.y_train(), lr_predictions_values.y_train()
            )
        ),
        {"size": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.6,
        str("Linear Regression Test"),
        {"size": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        str(
            classification_report(
                response_values.y_test(), lr_predictions_values.y_test()
            )
        ),
        {"size": 10},
        fontproperties="monospace",
    )
    plt.axis("off")
    plt.savefig(f"{BASE_PATH}/results/logistic_results.png")


def feature_importance_plot(model, x_data, output_pth):
    """
    creates and stores the feature importance in pth
    input:
            model: model object containing feature_importance
            x_data: pandas dataframe of X values
            output_pth: path to store the figure
    output:
             None
    """
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x_data)
    plt.figure(figsize=(15, 8))
    shap.summary_plot(values, x_data, plot_type="bar", show=False)
    plt.savefig(output_pth)


def train_models(x_train, x_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              model
    """
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(max_iter=10000)

    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_predictions_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_predictions_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_predictions_lr = lrc.predict(x_train)
    y_test_predictions_lr = lrc.predict(x_test)

    # plots
    plt.figure(figsize=(15, 8))
    axes = plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_, x_test, y_test, ax=axes, alpha=0.8)
    plot_roc_curve(lrc, x_test, y_test, ax=axes, alpha=0.8)
    plt.savefig(f"{BASE_PATH}/results/roc_curve_result.png")

    joblib.dump(cv_rfc.best_estimator_, "./models/rfc_model.pkl")
    joblib.dump(lrc, "./models/logistic_model.pkl")

    classification_report_image(
        YValues(y_train, y_test),
        YValues(y_train_predictions_lr, y_test_predictions_lr),
        YValues(y_train_predictions_rf, y_test_predictions_rf),
    )

    feature_importance_plot(
        cv_rfc.best_estimator_, x_train, f"{BASE_PATH}/results/feature_importances.png"
    )

    return cv_rfc.best_estimator_
