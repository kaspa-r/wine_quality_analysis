import statsmodels.api as sm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import List, Dict
import statsmodels.formula.api as smf
from itertools import combinations
import matplotlib.pyplot as plt


def reformating_columns(columns: List[str]) -> Dict:

    """
    Reformating the columns to be more legible for the data modelling process
    """

    dic = {}
    for col in columns:
        dic[col] = col.replace(' ', '_')
    return dic



def parameter_choice(parameter_space: List[str]) -> List:
    
    """
    Creates all possible combinations of independent variables for creating models
    """
    
    all_possible_combinations = list()
    for r in range(len(parameter_space)):
        all_possible_combinations.append([*combinations(parameter_space, r)])
    return all_possible_combinations


def setting_train_test_split(data : pd.DataFrame):
    
    """Setting up an 80-20 train-test split for data modelling"""

    train, test = train_test_split(data, train_size=0.8, random_state=2,
                                #    stratify=[''] --> look into this
                                   )
    
    #    class_weight --> not in train_test_split
    return train, test


def model_set_up(ind_variables : List[str], target : str, train : pd.DataFrame, test : pd.DataFrame, logistic : bool = False):
    
    """Setting up an OLS or an Ordinal Logistic Regression Model given the target and list of independent variables"""

    train_y = train[target]
    train_x = train[ind_variables]

    test_y = test[target]
    test_x = test[ind_variables]

    if logistic:
        formula = f"quality ~ {' + '.join(set(ind_variables))}"
        print(f"Testing logistic regression: {formula}")
        model = smf.mnlogit(formula, train, silent=True, random_state = 2)
        return model, test_x, test_y
    
    else:        
        x = sm.add_constant(train_x)
        model = sm.OLS(train_y, x, random_state = 2)
        return model, test_x, test_y
        

def calculate_metrics(model, variables : List[str], list_to_create : List, logistic = False, test_x = None, test_y = None) -> None:
    """
    Calculating relevant model performance metrics both for Regressions and Classification models
    
    For Classification models we calculate:
        1. Accuracy Score
        2. Precision Score
        3. Recall
        4. F1 Score
        5. ROC AUC score

    For Regression models we calculate:
        1. Adjusted R squared
        2. AIC
        3. BIC
        4. Log-Likelihood
        5. MSE

    The function also puts all data into the list_to_create
    """

    if logistic:

        test_x_const = sm.add_constant(test_x)
        pred_y = model.fit().predict(test_x_const).fillna(0)
        predictions = pred_y.idxmax(axis = 1)
        
        if isinstance(variables, list) or isinstance(variables, set):
            variables_string = ','.join(variables)
        list_to_create.loc[variables_string,'Ind_Variables'] = variables_string
        list_to_create.loc[variables_string,'Model'] = model
        list_to_create.loc[variables_string,'Accuracy'] = accuracy_score(test_y, predictions)
        list_to_create.loc[variables_string,'Precision'] = precision_score(test_y, predictions, average = 'weighted')
        list_to_create.loc[variables_string,'Recall'] = recall_score(test_y, predictions, average = 'weighted')
        list_to_create.loc[variables_string,'F1Score'] = f1_score(test_y, predictions, average = 'weighted')

    else:
        results = model.fit()

        if isinstance(variables, list):
            variables_string = ','.join(variables)
        list_to_create.loc[variables_string,'Ind_Variables'] = variables_string
        list_to_create.loc[variables_string,'Adjusted_RSquared'] = results.rsquared_adj
        list_to_create.loc[variables_string,'AIC'] = results.aic
        list_to_create.loc[variables_string,'BIC'] = results.bic
        list_to_create.loc[variables_string,'Model'] = model
        list_to_create.loc[variables_string,'Log-Likelihood'] = results.llf
        list_to_create.loc[variables_string,'MSE'] = results.mse_model

def remove_borders() -> None:
    """
    Removes borders from an existing graph.
    """
    for spine in plt.gca().spines.values():
        spine.set_visible(False)


def run_inferential_analysis_on_means(pd_series_1 : pd.DataFrame, pd_series_2 : pd.DataFrame, hypotheses_dictionary: Dict) -> None:
    """Does a T-test on the hypothesis (smaller, equal, or larger) of the means of two pd.Series"""

    for column in hypotheses_dictionary:
        hypothesis = hypotheses_dictionary[column]
        _, pvalue, _ = sm.stats.ttest_ind(pd_series_1[column], pd_series_2[column], alternative = hypothesis)
        print(f"""Hypothesis: {column} mean for exceptional quality wines is {hypothesis}\nP-value: {pvalue}\nVerdict: Statistically {pvalue < 0.05}""")

