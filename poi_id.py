#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

import warnings
warnings.filterwarnings(action='ignore')

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary', 'bonus',
                 'long_term_incentive', 'total_stock_value',
                 'exercised_stock_options',
                 'from_poi_to_this_person', 'from_this_person_to_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
def remove_outliers(data, outlier_keys):
    for outlier_key in outlier_keys:
        if outlier_key in data.keys():
            data.pop(outlier_key)

# remove outliers on salary
outliers = ['TOTAL', 'BANNANTINE JAMES M', 'GRAY RODNEY', 'WESTFAHL RICHARD K', 'DERRICK JR. JAMES V', 'WHALLEY LAWRENCE G', 'PICKERING MARK R', 'FREVERT MARK A']
remove_outliers(data_dict, outliers)

# remove outliers on bonus
outliers = ['LAVORATO JOHN J', 'ALLEN PHILLIP K', 'KITCHEN LOUISE', 'MCMAHON JEFFREY', 'FALLON JAMES B']
remove_outliers(data_dict, outliers)

# remove outliers on long_term_incentive
outliers = ['ECHOLS JOHN B', 'MARTIN AMANDA K']
remove_outliers(data_dict, outliers)

# remove outliers on total_stock_value
outliers = ['PAI LOU L', 'WHITE JR THOMAS E', 'BAXTER JOHN C', 'DIMICHELE RICHARD G', 'REDMOND BRIAN L', 'OVERDYKE JR JERE C', 'HORTON STANLEY C', 'ELLIOTT STEVEN']
remove_outliers(data_dict, outliers)

# remove outliers on from_this_person_to_poi
outliers = ['HAEDICKE MARK E', 'SHAPIRO RICHARD S', 'BUY RICHARD B', 'SHANKMAN JEFFREY A', 'KAMINSKI WINCENTY J', 'MCCONNELL MICHAEL S', 'BECK SALLY W', 'KEAN STEVEN J']
remove_outliers(data_dict, outliers)

### Task3: Create my own feature (PCA with deferral_payments and deferred_income)

# remove outliers on deferral_payments
outliers = ['HUMPHREY GENE E', 'MEYER ROCKFORD G']
remove_outliers(data_dict, outliers)

# remove outliers on deferred_income
outliers = ['MULLER MARK S', 'DETMERING TIMOTHY J']
remove_outliers(data_dict, outliers)

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def sort_without_nan(target_feature):
    tmp_data_dict = {key:val for key, val in data_dict.items() if val[target_feature] != 'NaN'}
    tmp_data_dict = sorted(tmp_data_dict.items(), key=lambda x: x[1][target_feature])
    tmp_data_dict = {item[0]:item[1] for item in tmp_data_dict}
    return tmp_data_dict

scaler = MinMaxScaler()
pca = PCA(n_components=1)

tmp_data_dict_deferral_payments = sort_without_nan('deferral_payments')
tmp_data_dict_deferred_income = sort_without_nan('deferred_income')

mean_deferral_payments = np.mean([val['deferral_payments'] for _, val in tmp_data_dict_deferral_payments.items()])
mean_deferred_income = np.mean([val['deferred_income'] for _, val in tmp_data_dict_deferred_income.items()])

for _, val in data_dict.items():
    if val['deferral_payments'] == 'NaN':
        val['deferral_payments'] = mean_deferral_payments

for _, val in data_dict.items():
    if val['deferred_income'] == 'NaN':
        val['deferred_income'] = mean_deferred_income

tmp_deferral_payments = np.array([val['deferral_payments'] for _, val in data_dict.items()]).reshape(-1, 1)
tmp_deferred_income = np.array([val['deferred_income'] for _, val in data_dict.items()]).reshape(-1, 1)


tmp_deferral_payments = scaler.fit_transform(tmp_deferral_payments)
tmp_deferred_income = scaler.fit_transform(tmp_deferred_income)

X = [[tmp_deferral_payments[idx][0], tmp_deferred_income[idx][0]] for idx in range(len(tmp_deferral_payments))]
X = pca.fit_transform(X)

for idx, key in enumerate(data_dict):
    data_dict[key]['deferral_payments+deferred_income'] = X[idx][0]

features_list.append('deferral_payments+deferred_income')

outliers = ['WASAFF GEORGE', 'GAHN ROBERT S']
remove_outliers(data_dict, 'deferral_payments+deferred_income')

my_dataset = data_dict #data_dict

### Extract features and labels from dataset for local testing
features_list = ['poi', 'bonus']

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(algorithm='SAMME.R', learning_rate=0.5, n_estimators=30, random_state=42)
test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


dump_classifier_and_data(clf, my_dataset, features_list)
