# author: Noel Kronenberg

import corr_utils
import corr_utils.covariate

import pandas as pd
from enum import Enum
import numpy as np
import operator
from scipy.stats import norm

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from dcurves import dca, plot_graphs

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_selection import RFE

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline

plt.rcParams['font.family'] = 'DejaVu Sans'
np.random.seed(42)

global default_key 
# default_key:str = corr_utils.default_key

def evaluate_subgroups(subgroups:dict, score_columns:list, outcome_column:str, test_size=0.2, dca_y_limits:list=[-0.01, 0.02], dca_thresholds:np.ndarray=np.arange(0, 0.10, 0.01), categorical_columns:list=[], calculate_proba:bool=False) -> None:
    """
    Validates a score on all important metrics.

    Parameters:
        subgroups (dict): A dict of subgroup DataFrames ('name': DataFrame). Works with create_subgroups().
        score_columns (list): The column names containing the scores.
        outcome_column (str): The column name containing the outcome indicators.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        dca_y_limits (list, optional): Y-axis limits for the decision curve analysis (DCA) plot. Defaults to [-0.01, 0.02].
        dca_thresholds (np.ndarray, optional): Array of thresholds for DCA. Defaults to np.arange(0, 0.10, 0.01).
        categorical_columns (list): A list of columns to consider as categorical (will be one-hot encoded). Defaults to [].
        calculate_proba(bool, optional): Whether to calculate outcome probabilities for scores (on test data). Defaults to False. 

    Returns:
        None: Prints the results
    """
     
    for subgroup in subgroups.keys():
        print('########################################')
        print(subgroup, '\n')
        validate_score(subgroups[subgroup], score_columns, outcome_column, test_size, dca_y_limits, dca_thresholds, categorical_columns=categorical_columns, calculate_proba=calculate_proba)

def get_probabilities(df:pd.DataFrame, score_column:str, outcome_column:str, plot:bool=False, categorical_columns:list=[], show_regression:bool=False) -> pd.DataFrame:
    """
    Calculate the probabilities for an outcome given a score value.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.
        score_column (str): The column name containing the scores.
        outcome_column (str): The column name containing the outcome indicators.
        plot (bool): Plot the resulting probabilities. Defaults to False.
        categorical_columns (list): A list of columns to consider as categorical (will be one-hot encoded). Defaults to [].
        show_regression (bool, optional): Whether to show the regression results. Defaults to False.
    
    Returns:
        pd.DataFrame: A DataFrame containing the probabilities.
    """

    df_temp = df.copy()
      
    X = prepare_regression_data(df_temp[[score_column]], scale=False, intercept=True, categorical_columns=categorical_columns)
    y = df_temp[outcome_column]

    model = sm.Logit(y, X)
    result = model.fit()

    if show_regression:
        print(result.summary())

    unique_scores = np.sort(df_temp[score_column].unique()).flatten()

    if categorical_columns != []:
        unique_scores_encoded = pd.get_dummies(unique_scores, drop_first=True)
        X_new = sm.add_constant(unique_scores_encoded) 
        predicted_proba = result.predict(X_new).values.ravel()
    else:
        X_new = sm.add_constant(unique_scores) 
        predicted_proba = result.predict(X_new)    
    
    df_probabilities = pd.DataFrame({score_column: unique_scores, f'{score_column}_probability': predicted_proba})

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(df_probabilities[score_column], df_probabilities[f'{score_column}_probability'], marker='o', linestyle='-', color='black')
        plt.title(f'{outcome_column}_probability vs. {score_column}')
        plt.xlabel(score_column)
        plt.ylabel(f'{score_column}_probability')
        plt.grid(True)
        plt.show()

    return df_probabilities

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_test_data(df:pd.DataFrame, score_column:str='', outcome_column:str='', test_size:float=0.2, confounders:list=[], categorical_columns:list=[], show_regression:bool=False, calculate_proba:bool=False) -> pd.DataFrame:
    """
    Calculate the probabilities for an outcome given a score value and return a test set that was excluded in that calculation. Allows for confounders to be included.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data. 
        score_column (str): The column name containing the scores. Defaults to ''.
        outcome_column (str): The column name containing the outcome indicators. Defaults to ''.
        test_size (str, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        confounders (list, optional): A list of confounders to include. Defaults to [].
        categorical_columns (list): A list of columns to consider as categorical (will be one-hot encoded). Defaults to [].
        show_regression (bool, optional): Whether to show the regression results. Defaults to False.
        calculate_proba(bool, optional): Whether to calculate outcome probabilities for scores (on test data). Defaults to False. 
    
    Returns:
        pd.DataFrame: A DataFrame containing a test set with the probabilities.
    """

    df_temp = df.copy()

    if test_size != 0:
        df_train, df_test = train_test_split(df_temp, test_size=test_size, random_state=42)
    else:
        df_train = df_test = df_temp

    if calculate_proba:
        if confounders == []:
            df_probabilities = get_probabilities(df_train, score_column=score_column, outcome_column=outcome_column, categorical_columns=categorical_columns, show_regression=show_regression)
            df_test = pd.merge(df_test[[score_column, outcome_column]], df_probabilities, on=score_column, how='left')
        else:
            X = prepare_regression_data(df_train[[score_column] + confounders], scale=False, intercept=True, categorical_columns=categorical_columns)
            y = df_train[outcome_column]

            model = sm.Logit(y, X)
            result = model.fit()

            if show_regression:
                print(result.summary())

            X_test = prepare_regression_data(X_test[[score_column] + confounders], scale=False, intercept=True, categorical_columns=categorical_columns)
            df_test[f'{score_column}_probability'] = result.predict(X_test)

            df_test = df_test[[score_column] + confounders + [f'{score_column}_probability'] + [outcome_column]]

    return df_test

def get_train_data(df:pd.DataFrame, score_column:str='', outcome_column:str='', test_size:float=0.2, confounders:list=[], categorical_columns:list=[], show_regression:bool=False, calculate_proba:bool=False) -> pd.DataFrame:
    """
    Calculate the probabilities for an outcome given a score value and return a train set that was included in that calculation. Allows for confounders to be included.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data. 
        score_column (str): The column name containing the scores. Defaults to ''.
        outcome_column (str): The column name containing the outcome indicators. Defaults to ''.
        test_size (str, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        confounders (list, optional): A list of confounders to include. Defaults to [].
        categorical_columns (list): A list of columns to consider as categorical (will be one-hot encoded). Defaults to [].
        show_regression (bool, optional): Whether to show the regression results. Defaults to False.
        calculate_proba(bool, optional): Whether to calculate outcome probabilities for scores (on test data). Defaults to False. 
    
    Returns:
        pd.DataFrame: A DataFrame containing a train set with the probabilities.
    """

    df_temp = df.copy()

    if test_size != 0:
        df_train, _ = train_test_split(df_temp, test_size=test_size, random_state=42)
    else:
        df_train = df_temp

    if calculate_proba:
        if confounders == []:
            df_probabilities = get_probabilities(df_train, score_column=score_column, outcome_column=outcome_column, categorical_columns=categorical_columns, show_regression=show_regression)
            df_train = pd.merge(df_train, df_probabilities, on=score_column, how='left')
        else:
            X = prepare_regression_data(df_train[[score_column] + confounders], scale=False, intercept=True, categorical_columns=categorical_columns)
            y = df_train[outcome_column]

            model = sm.Logit(y, X)
            result = model.fit()

            if show_regression:
                print(result.summary())

            df_train[f'{score_column}_probability'] = result.predict(X)

    return df_train

def get_confidence_intervals(df:pd.DataFrame, score_column:str, outcome_column:str, test_size:float=0.2, confounders:list=[], categorical_columns:list=[], show_regression:bool=False) -> pd.DataFrame:
    """
    Calculate the probabilities confidence intervals (CI) around the probabilities for a score and outcome and modifies the DataFrame in place (!). Allows for confounders to be included.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data. 
        score_column (str): The column name containing the scores. 
        outcome_column (str): The column name containing the outcome indicators. 
        test_size (str, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        confounders (list, optional): A list of confounders to include. Defaults to [].
        categorical_columns (list): A list of columns to consider as categorical (will be one-hot encoded). Defaults to [].
        show_regression (bool, optional): Whether to show the regression results. Defaults to False.
    
    Returns:
        None: The original DataFrame is modified in place (!).
    """

    if test_size != 0:
        df_train, _ = train_test_split(df, test_size=test_size, random_state=42)
    else:
        df_train = df
    
    if confounders == []:
        X = prepare_regression_data(df_train[[score_column]], scale=False, intercept=True, categorical_columns=categorical_columns)
    else:
        X = prepare_regression_data(df_train[[score_column] + confounders], scale=False, intercept=True, categorical_columns=categorical_columns)
    
    y = df_train[outcome_column]

    model = sm.Logit(y, X).fit()

    if show_regression:
        print(model.summary())

    # df_train_full = df_train.copy()
    # df_train_full[f'{score_column}_probability'] = predicted_probs = model.predict(X)

    predicted_probs = model.predict(X)

    # based on: https://stackoverflow.com/a/47419474; reference: https://en.wikipedia.org/wiki/Delta_method
    cov = model.cov_params()
    gradient = (predicted_probs * (1 - predicted_probs) * X.T).T 
    gradient = np.array(gradient, dtype=float)
    std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov), g)) for g in gradient])
    z = norm.ppf(0.975) # z-score for 95% CI
    ci_upper = np.maximum(0, np.minimum(1, predicted_probs + std_errors * z))
    ci_lower = np.maximum(0, np.minimum(1, predicted_probs - std_errors * z))

    df_train[f'{score_column}_probability_CI_lower'] = ci_lower
    df_train[f'{score_column}_probability_CI_upper'] = ci_upper

    ci_map_lower = df_train[[score_column, f'{score_column}_probability_CI_lower']].drop_duplicates().set_index(score_column).to_dict()[f'{score_column}_probability_CI_lower']
    ci_map_upper = df_train[[score_column, f'{score_column}_probability_CI_upper']].drop_duplicates().set_index(score_column).to_dict()[f'{score_column}_probability_CI_upper']

    df[f'{score_column}_probability_CI_lower'] = df[score_column].map(ci_map_lower)
    df[f'{score_column}_probability_CI_upper'] = df[score_column].map(ci_map_upper)

def get_probabilities_for_cohort(df:pd.DataFrame, score_column:str, outcome_column:str, test_size:float=0.2, confounders:list=[], categorical_columns:list=[], show_regression:bool=False) -> pd.DataFrame:
    """
    Calculate the probabilities for an outcome given a score value and modify DataFrame in place (!). Allows for confounders to be included.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.
        score_column (str): The column name containing the scores.
        outcome_column (str): The column name containing the outcome indicators.
        test_size (str, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        confounders (list, optional): A list of confounders to include. Defaults to [].
        categorical_columns (list): A list of columns to consider as categorical (will be one-hot encoded). Defaults to [].
        show_regression (bool, optional): Whether to show the regression results. Defaults to False.
    
    Returns:
        None: The original DataFrame is modified in place (!).
    """

    df_train = get_train_data(df, score_column=score_column, outcome_column=outcome_column, test_size=test_size, confounders=confounders, categorical_columns=categorical_columns, show_regression=show_regression, calculate_proba=True)
    # df_test = get_test_data(df_temp, test_siz=test_size)

    proba_map = df_train[[score_column, f'{score_column}_probability']].drop_duplicates().set_index(score_column).to_dict()[f'{score_column}_probability']

    df[f'{score_column}_probability'] = df[score_column].map(proba_map)

def get_auroc(df:pd.DataFrame, score_columns:list, outcome_column:str, test_size:float=0.2, categorical_columns:list=[], calculate_proba:bool=False) -> None:
    """
    Plots the ROC curve with AUC for a score and outcome.

    Parameters:
        score_columns (list): The column names containing the scores.
        outcome_column (str): The column name containing the outcome indicators.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        categorical_columns (list): A list of columns to consider as categorical (will be one-hot encoded). Defaults to [].
        calculate_proba(bool, optional): Whether to calculate outcome probabilities for scores (on test data). Defaults to False.

    Returns:
        None
    """

    plt.figure(figsize=(10, 8))

    for score_column in score_columns:
        df_test = get_test_data(df, score_column=score_column, outcome_column=outcome_column, test_size=test_size, categorical_columns=categorical_columns, calculate_proba=calculate_proba) # NOTE: test_size == 0 is handled in get_test_data

        y_true = df_test[outcome_column]
        y_score = df_test[f'{score_column}_probability']

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, label=f'{score_column} (AUC = {auc:.4f})')

    plt.plot([0, 1], [0, 1], color='black', linestyle='--') # baseline
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# reference: https://glassboxmedicine.com/2019/03/02/measuring-performance-auprc/
def get_auprc(df: pd.DataFrame, score_columns: list, outcome_column: str, test_size: float = 0.2, categorical_columns:list=[], calculate_proba:bool=False) -> None:
    """
    Plots the Precision-Recall curve (PRC) with AUC for a score and outcome.

    Parameters:
        score_columsn (list): The column names containing the scores.
        outcome_column (str): The column name containing the outcome indicators.
        label_name (str): Name of the model or label for legend.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        categorical_columns (list): A list of columns to consider as categorical (will be one-hot encoded). Defaults to [].
        calculate_proba(bool, optional): Whether to calculate outcome probabilities for scores (on test data). Defaults to False. 

    Returns:
        None
    """

    plt.figure(figsize=(10, 8))

    for score_column in score_columns:
        df_test = get_test_data(df, score_column=score_column, outcome_column=outcome_column, test_size=test_size, categorical_columns=categorical_columns, calculate_proba=calculate_proba)  # NOTE: test_size == 0 is handled in get_test_data

        y_true = df_test[outcome_column]
        y_score = df_test[score_column]

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auc_pr = average_precision_score(y_true, y_score)

        # baseline (ratio of positive instances)
        baseline = sum(y_true) / len(y_true)

        plt.plot(recall, precision, label=f'{score_column} (AUPRC = {auc_pr:.4f})')    

    plt.axhline(y=baseline, color='black', linestyle='--', label=f'Baseline = {baseline:.4f}') # baseline
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def prepare_regression_data(df_input_data:pd.DataFrame, scale:bool=True, intercept:bool=True, categorical_columns:list=[]) -> pd.DataFrame:
    """
    Prepares data for regression. Adds an intercept and optionally scales the data and also one-hot encodes categorical variables.
    
    Parameters:
        df_input_data (pandas.DataFrame): DataFrame containing the data.
        scale (bool, optional): If True, data will be scaled (range(0,1)). Defaults to True.
        intercept (bool, optional): If True, an intercept will be added. Defaults to True.
        categorical_columns (list, optional): Columns containing categorical data (will be one-hot encoded). Defaults to [].

    Returns:
        pandas.DataFrame: DataFrame with the prepared data.
    """
      
    # one-hot encoding for categorical columns
    df_data = pd.get_dummies(df_input_data.copy(), columns=categorical_columns, drop_first=True)

    # scale data
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        ndarray_data = scaler.fit_transform(df_data)
        df_data = pd.DataFrame(ndarray_data, columns=df_data.columns, index=df_data.index) # convert back to df

    # add intercept
    if intercept:
        df_data = sm.add_constant(df_data)

    return df_data

class Model(Enum):
    """
    Enumeration for selecting different model packages (SKLEARN = sklearn for prediction tasks, STATS = statsmodels for a more statistical approach).
    """

    SKLEARN = 1
    STATS = 2

def get_regression(X:pd.DataFrame, y:pd.Series, test_size:float=0.2, label_name:str='', categorical_columns=[], use_lasso:bool=False, intercept:bool=True, scale_data:bool=True, selected_model:Model=Model.STATS) -> pd.DataFrame:
    """
    Trains a logistic regression model on the given data. Plots the results.
    
    Parameters:
        X (pandas.DataFrame): Features for training the model.
        y (pandas.Series): Target variable.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        label_name (str, optional): Label for title and legend.
        categorical_columns (list): A list of columns to consider as categorical (will be one-hot encoded). Defaults to [].
        use_lasso (bool, optional): Regularize the regression with lasso. Defaults to False.
        intercept (bool, optional): If True, an intercept will be added. Defaults to True.
        scale_data (bool, optional): If True, data will be scaled (range(0,1)). Defaults to True.
        selected_model (Model, optional): Selection of which regression model to use (Model.SKLEARN = sklearn for prediction tasks, Model.STATS = statsmodels for a more statistical approach). Defaults to STATS.

    Returns:
        pandas.DataFrame: A DataFrames for the weights.
    """

    # data preparation

    X = prepare_regression_data(X, scale=scale_data, intercept=intercept, categorical_columns=categorical_columns)

    if test_size != 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else: # NOTE: check sensibility
        X_train = X_test = X
        y_train = y_test = y

    # training

    if selected_model == Model.SKLEARN:
        if use_lasso:
                model = LogisticRegression(penalty='l1', solver='saga', random_state=42, max_iter=10000, C=1.0)
        else:
            model = LogisticRegression(random_state=42)

        pipe = make_pipeline(model) # apply scaling on training data (reference: https://scikit-learn.org/stable/modules/preprocessing.html)
        pipe.fit(X_train, y_train) 
        y_pred_proba = pipe.predict_proba(X_test)[:, 1] # positive class probabilities
    
    if selected_model == Model.STATS:
        model = sm.Logit(y_train, X_train)
        if use_lasso:
            result = model.fit_regularized(method='l1', alpha=1.0)
        else:
            result = model.fit()

        y_pred_proba = result.predict(X_test)

    # results

    if selected_model == Model.STATS:

        print(result.summary())

        # odds ratios (reference: https://stackoverflow.com/a/47740828)

        params = result.params
        odds_ratios = result.conf_int()
        odds_ratios['Odds Ratio'] = params
        odds_ratios.columns = ['5%', '95%', 'Odds Ratio']
        print(np.exp(odds_ratios))

        # AU-ROC

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.plot(fpr, tpr, label=f'{label_name} Logistic Regression (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.show()

        # AU-PRC

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auc_pr = average_precision_score(y_test, y_pred_proba)

        # baseline (ratio of positive instances)
        baseline = sum(y_test) / len(y_test)

        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label=f'{label_name} (AUPRC = {auc_pr:.4f})')    
        plt.axhline(y=baseline, color='black', linestyle='--', label=f'Baseline = {baseline:.4f}') # baseline
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

        # weights

        feature_names = X.columns
        weights = result.params.values.flatten()
        df_weights = pd.DataFrame({'feature': feature_names, 'weight': weights})
        df_weights = df_weights.sort_values(by='weight', ascending=True)

        plt.figure(figsize=(10, 8))
        plt.barh(df_weights['feature'], df_weights['weight'])
        plt.xlabel('Weight')
        plt.ylabel('Feature')
        plt.title(f'{label_name} Feature Weights')
        plt.grid(True, axis='x', linestyle='--', alpha=0.5)
        plt.show()

        # p-values

        p_values = result.pvalues
        df_p_values = pd.DataFrame({'feature': feature_names, 'p-value': p_values})
        df_p_values = df_p_values.sort_values(by='p-value')
        df_p_values['significant (< 0.05)'] = df_p_values['p-value'] < 0.05

        print(df_p_values)

    if selected_model == Model.SKLEARN:
        # AU-ROC

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.plot(fpr, tpr, label=f'{label_name} Logistic Regression (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.show()

        # weights

        feature_names = X.columns
        weights = pipe.named_steps['logisticregression'].coef_.flatten()
        df_weights = pd.DataFrame({'feature': feature_names, 'weight': weights})
        df_weights = df_weights.sort_values(by='weight', ascending=True)

        plt.figure(figsize=(10, 8))
        plt.barh(df_weights['feature'], df_weights['weight'])
        plt.xlabel('Weight')
        plt.ylabel('Feature')
        plt.title(f'{label_name} Feature Weights')
        plt.grid(True, axis='x', linestyle='--', alpha=0.5)
        plt.show()

    return df_weights

def normalize_weights(df:pd.DataFrame, weight_column:str='weight', threshold:float=0.05, save_to:str=None) -> pd.DataFrame:
    """
    Normalize the weights in a DataFrame relative to the smallest non-zero absolute weight and set near-zero values to 0.
    
    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the weights and variables.
        weight_column (str, optional): Column inside the DataFrame containing the weights. Defaults to 'weight'.
        threshold (float, optional): Threshold for when to consider a value to be 0. Defaults to 0.05.
        save_to (str: optional): Path to where a csv of the weights should be saved. Defaults to None.

    Returns:
        pandas.DataFrame: The DataFrame with the normalized weights.
    """

    df_temp = df.copy()

    df_temp.loc[df_temp[weight_column].abs() < threshold, weight_column] = 0

    # smallest non-zero absolute weight
    min_weight = df_temp[df_temp[weight_column] != 0][weight_column].abs().min()
    df_temp['normalized_weight'] = df_temp[weight_column] / min_weight

    if save_to != None:
        df_temp.to_csv(save_to, index=False)

    df_temp.drop(columns=[weight_column], inplace=True)

    return df_temp

def load_weights(df:pd.DataFrame, features:list, feature_column:str='feature', weight_column:str='normalized_weight') -> dict:
    """
    Loads weights for specified features from a DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with 'feature' and 'normalized_weight' columns.
        features (list): Features to load weights for.
        feature_column (str): Column name with the feature names. Defaults to 'feature'.
        weight_column (str): Column name with the weights. Defaults to 'normalized_weight'.

    Returns:
        dict: Feature names mapped to their weights. Unfound features get `None`.
    """

    weights = {}
    for feature in features:
        try:
            weight = float(df[df[feature_column] == feature][weight_column])
            weights[feature] = weight
        except KeyError:
            print(f"Warning: {feature} not found in DataFrame.")
            weights[feature] = None

    return weights

def get_brier(df:pd.DataFrame, score_column:str, outcome_column:str, test_size=0.2, categorical_column:list=[], calculate_prob:bool=False) -> None:
    """
    Calculate and print the Brier score for a score and outcome.
    
    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.
        score_column (str): The column name containing the scores.
        outcome_column (str): The column name containing the outcome indicators.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        categorical_column (list): Column to consider as categorical (will be one-hot encoded). Defaults to [].
        calculate_proba(bool, optional): Whether to calculate outcome probabilities for scores (on test data). Defaults to False. 
    
    Returns:
        None
    """

    df_test = get_test_data(df, score_column=score_column, outcome_column=outcome_column, test_size=test_size, categorical_columns=categorical_column, calculate_proba=calculate_prob) # NOTE: test_size == 0 is handled in get_test_data

    if df_test[f'{score_column}_probability'].isnull().sum() > 0:
        print(f'Probabilities for {score_column} contain missings.')
        df_test = corr_utils.covariate.exclude_rows(df_test, f'{score_column}_probability', [np.nan]) # NOTE: investigate missings

    # reference: https://scikit-learn.org/stable/modules/model_evaluation.html#brier-score-loss
    y_true = df_test[outcome_column].astype(int)
    y_prob = df_test[f'{score_column}_probability']
    brier_score = brier_score_loss(y_true, y_prob)

    print(f'Brier score for {score_column} and {outcome_column}: {brier_score}')

    return brier_score

def get_brier_skill(df:pd.DataFrame, score_column:str, outcome_column:str, test_size:float=0.2, categorical_column:list=[], calculate_proba:bool=False) -> float:
    """
    Calculate and print the Brier skill score for a score and outcome.
    
    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.
        score_column (str): The column name containing the scores.
        outcome_column (str): The column name containing the outcome indicators.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        categorical_column (list): Column to consider as categorical (will be one-hot encoded). Defaults to [].
        calculate_proba(bool, optional): Whether to calculate outcome probabilities for scores (on test data). Defaults to False. 
    
    Returns:
        float: The Brier skill score.
    """

    df_test = get_test_data(df, score_column=score_column, outcome_column=outcome_column, test_size=test_size, categorical_columns=categorical_column, calculate_proba=calculate_proba) # NOTE: test_size == 0 is handled in get_test_data

    if df_test[f'{score_column}_probability'].isnull().sum() > 0:
        print(f'Probabilities for {score_column} contain missings.')
        df_test = corr_utils.covariate.exclude_rows(df_test, f'{score_column}_probability', [np.nan]) # NOTE: investigate missings

    # reference: https://scikit-learn.org/stable/modules/model_evaluation.html#brier-score-loss
    y_true = df_test[outcome_column].astype(int)
    y_prob = df_test[f'{score_column}_probability']
    brier_score = brier_score_loss(y_true, y_prob)
    print(f'Brier score for {score_column} and {outcome_column}: {brier_score}')

    # reference / baseline score (reference: https://machinelearningmastery.com/probability-metrics-for-imbalanced-classification/)
    baseline = np.mean(df[outcome_column])
    baseline_brier = brier_score_loss(y_true, np.full_like(y_true, baseline, dtype=float))
    print(f'Baseline Brier score: {baseline_brier}')

    skill_score = 1.0 - (brier_score / baseline_brier)
    print(f'Brier skill score: {skill_score}')

    return skill_score

def get_calibration(df:pd.DataFrame, score_columns:list, outcome_column:str, test_size:float=0.2, groups:int=10, categorical_columns:list=[], calculate_proba:bool=False) -> None:
    """ 
    Plots the calibration plot for a given score and outcome. Also prints the calibration slope and intercept.
    
    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.
        score_columns (list): The column names containing the scores.
        outcome_column (str): The column name containing the outcome indicators.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        groups (int, optional): Number of groups to split the data into. Defaults to 10.
        categorical_columns (list): A list of columns to consider as categorical (will be one-hot encoded). Defaults to [].
        calculate_proba(bool, optional): Whether to calculate outcome probabilities for scores (on test data). Defaults to False. 
    
    Returns:
        None
    """

    plt.figure(figsize=(10, 6))

    for score_column in score_columns:
        df_test = get_test_data(df, score_column=score_column, outcome_column=outcome_column, test_size=test_size, categorical_columns=categorical_columns, calculate_proba=calculate_proba) # NOTE: test_size == 0 is handled in get_test_data
        
        observed = df_test[outcome_column]
        predicted = df_test[f'{score_column}_probability']

        prob_true, prob_pred = calibration_curve(observed, predicted, n_bins=groups)
        plt.plot(prob_pred, prob_true, 's-', label=f'Calibration for {score_column}')

    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot')
    plt.legend()
    plt.show()

    if not (np.isnan(prob_pred).any() or np.isnan(prob_true).any()): 
        model = LinearRegression().fit(np.array(prob_pred).reshape(-1, 1), prob_true)
        calibration_slope = model.coef_[0]
        calibration_intercept = model.intercept_
    else:
        calibration_slope = np.nan
        calibration_intercept = np.nan

    print(f'Calibration slope: {calibration_slope}')
    print(f'Calibration intercept: {calibration_intercept}')

def get_score_regression(df:pd.DataFrame, score_column:str, outcome_column:str) -> None:
    """
    Performs logistic regression for a categorical score and its predicted outcome. Prints the results.
    
    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.
        score_column (str): The column name containing the scores.
        outcome_column (str): The column name containing the outcome indicators.
    
    Returns:
        None
    """
      
    # reference: https://mskcc-epi-bio.github.io/decisioncurveanalysis/dca-tutorial-python.html#Multivariable_Decision_Curve_Analysis
    model = sm.GLM.from_formula(f'{outcome_column} ~ C({score_column})', data=df, family=sm.families.Binomial())
    results = model.fit()

    print(results.summary())

def get_dca(df:pd.DataFrame, score_columns:list, outcome_column:str, test_size:float=0.2, y_limits:list=[-0.01, 0.02], thresholds:np.ndarray=np.arange(0, 0.10, 0.01), categorical_columns:list=[], calculate_proba:bool=False) -> pd.DataFrame:
    """
    Plot decision curves for a given score and outcome.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.
        score_columns (list): The column names containing the scores.
        outcome_column (str): The column name containing the outcome indicators.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        y_limits (list, optional): Y-axis limits for the plot. Defaults to [-0.01, 0.02].
        thresholds (np.ndarray, optional): Array of thresholds for decision curve analysis. Defaults to np.arange(0, 0.10, 0.01).
        categorical_columns (list): A list of columns to consider as categorical (will be one-hot encoded). Defaults to [].
        calculate_proba(bool, optional): Whether to calculate outcome probabilities for scores (on test data). Defaults to False. 

    Returns:
        pd.DataFrame: A DataFrame containing the results.
    """

    categorical_column = []
    if score_columns[0] in categorical_columns:
        categorical_column = [score_columns[0]]      

    df_test = get_test_data(df, score_column=score_columns[0], outcome_column=outcome_column, test_size=test_size, categorical_columns=categorical_column, calculate_proba=calculate_proba) # NOTE: test_size == 0 is handled in get_test_data
    df_test.drop(columns=score_columns[0], inplace=True)
    df_test.rename(columns={f'{score_columns[0]}_probability':f'{score_columns[0]}'}, inplace=True)

    if len(score_columns) > 1:
        for score_column in score_columns[1:]:
            categorical_column = []
            if score_column in categorical_columns:
                categorical_column = [score_column]             
            df_test[score_column] = get_test_data(df, score_column, outcome_column, test_size, categorical_columns=categorical_column, calculate_proba=calculate_proba)[f'{score_column}_probability'] 

    # reference: https://mskcc-epi-bio.github.io/decisioncurveanalysis/dca-tutorial-python.html
    df_dca = dca(data=df_test, outcome=outcome_column, modelnames=score_columns, thresholds=thresholds)
    
    # print(df_dca)
    plot_graphs(plot_df=df_dca, graph_type='net_benefit', y_limits=y_limits, color_names=['blue', 'green', 'red', 'pink', 'black'][:len(score_columns)+2])

    return df_dca

def validate_score(df:pd.DataFrame, score_columns:list, outcome_column:str, test_size:float=0.2, dca_y_limits:list=[-0.01, 0.02], dca_thresholds:np.ndarray=np.arange(0, 0.10, 0.01), categorical_columns:list=[], calculate_proba:bool=False):
    """
    Validates a score on all important metrics.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.
        score_columns (list): The column names containing the scores.
        outcome_column (str): The column name containing the outcome indicators.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        dca_y_limits (list, optional): Y-axis limits for the decision curve analysis (DCA) plot. Defaults to [-0.01, 0.02].
        dca_thresholds (np.ndarray, optional): Array of thresholds for DCA. Defaults to np.arange(0, 0.10, 0.01).
        categorical_columns (list): A list of columns to consider as categorical (will be one-hot encoded). Defaults to [].
        calculate_proba(bool, optional): Whether to calculate outcome probabilities for scores (on test data). Defaults to False. 

    Returns:
        None: Prints the results
    """

    print('####################')
    print('Performance:\n')

    for score_column in score_columns:
        categorical_column = []
        if score_column in categorical_columns:
            categorical_column = [score_column]
        _ = get_brier_skill(df=df, score_column=score_column, outcome_column=outcome_column, test_size=test_size, categorical_column=categorical_column, calculate_proba=calculate_proba)

    print('\n')
    print('####################')
    print('Calibration: \n')

    _ = get_calibration(df=df, score_columns=score_columns, outcome_column=outcome_column, test_size=test_size, categorical_columns=categorical_columns, calculate_proba=calculate_proba)

    print('####################')
    print('Discrimination: \n')

    _ = get_auroc(df=df, score_columns=score_columns, outcome_column=outcome_column, test_size=test_size, categorical_columns=categorical_columns, calculate_proba=calculate_proba)
    _ = get_auprc(df=df, score_columns=score_columns, outcome_column=outcome_column, test_size=test_size, categorical_columns=categorical_columns, calculate_proba=calculate_proba)

    print('####################')
    print('Clinical Value: \n')

    _ = get_dca(df=df, score_columns=score_columns, outcome_column=outcome_column, test_size=test_size, y_limits=dca_y_limits, thresholds=dca_thresholds, categorical_columns=categorical_columns, calculate_proba=calculate_proba)

def get_important_features(df:pd.DataFrame, feature_column:str, outcome_column:str, n_features:int=10) -> list:
    """
    Gets the most important features predicting an outcome by means of Recursive Feature Elimination (RFE).

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.
        feature_column (list): The column name containing the feature.
        outcome_column (str): The column name containing the outcome indicators.
        n_features (int, optional): The amount of features to select. Defaults to 10.

    Returns:
        list: A list containing the selected features. Also prints AU-ROC.
    """

    # convert to one-hot encoded variables; split data
    X = pd.get_dummies(df[feature_column])
    y = df[outcome_column]

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # feature selection using RFE (reference: https://www.geeksforgeeks.org/recursive-feature-elimination-with-cross-validation-in-scikit-learn/)
    model = LogisticRegression(max_iter=1000, random_state=42)
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    rfe = rfe.fit(X_train, y_train)

    # get selected features (reference: https://stackoverflow.com/a/51090544)
    selected_features = X.columns[rfe.support_]
    print(f'Selected ICD code features: {selected_features}')

    # build model
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    model.fit(X_train_selected, y_train)

    # evaluate model
    y_pred = model.predict(X_test_selected)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_selected)[:, 1])
    print(f'AUC-ROC: {roc_auc}')

    return list(selected_features)

def convert_score(df:pd.DataFrame, original_score:str, new_score:str) -> pd.DataFrame:
    """
    Convert a score back to the same values as the original by building groups.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.
        original_score (list): The column names containing the old scores.
        new_score (str):  The column names containing the new scores.

    Returns:
        pd.DataFrame: The DataFrame with the added converted score.
    """

    df_temp = df.copy()

    # old_min = int(df_temp[original_score].min())
    # old_max = int(df_temp[original_score].max())
    old_length = df_temp[original_score].nunique() 
    old_labels = sorted(df_temp[original_score].unique())

    unique_scores = sorted(df_temp[new_score].unique())

    new_score_groups = pd.qcut(unique_scores, q=old_length, labels=old_labels)
    group_map = {score: group for score, group  in zip(unique_scores, new_score_groups)}

    df_temp[f'{new_score}_converted'] = df_temp[new_score].map(group_map)

    return df_temp