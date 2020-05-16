# %%
import os
import sys
from io import StringIO # Python 3.x
import fnmatch
import pandas as pd
import numpy as np
import statistics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import category_encoders as ce
import math
from collections import Counter
import scipy.stats as ss

# Missing Imputation
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.experimental import enable_iterative_imputer
from sklearn.compose import ColumnTransformer

from fancyimpute import IterativeImputer

# Outlier Detection Library
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.utils.data import generate_data, get_outliers_inliers

# AWS
import boto3

_REPLACE = 'replace'
_DROP = 'drop'
_DROP_SAMPLES = 'drop_samples'
_DROP_FEATURES = 'drop_features'
_SKIP = 'skip'
_DEFAULT_REPLACE_VALUE = 0.0

AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')


def convert(data, to):
    converted = None
    if to == 'array':
        if isinstance(data, np.ndarray):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values
        elif isinstance(data, list):
            converted = np.array(data)
        elif isinstance(data, pd.DataFrame):
            converted = data.as_matrix()
    elif to == 'list':
        if isinstance(data, list):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values.tolist()
        elif isinstance(data, np.ndarray):
            converted = data.tolist()
    elif to == 'dataframe':
        if isinstance(data, pd.DataFrame):
            converted = data
        elif isinstance(data, np.ndarray):
            converted = pd.DataFrame(data)
    else:
        raise ValueError("Unknown data conversion: {}".format(to))
    if converted is None:
        raise TypeError(
            'cannot handle data conversion of type: {} to {}'.format(
                type(data), to))
    else:
        return converted


def replace_nan_with_value(x, y, value):
    x = np.array([v if v == v and v is not None else value for v in x])  # NaN != NaN
    y = np.array([v if v == v and v is not None else value for v in y])
    return x, y


def remove_incomplete_samples(x, y):
    x = [v if v is not None else np.nan for v in x]
    y = [v if v is not None else np.nan for v in y]
    arr = np.array([x, y]).transpose()
    arr = arr[~np.isnan(arr).any(axis=1)].transpose()
    if isinstance(x, list):
        return arr[0].tolist(), arr[1].tolist()
    else:
        return arr[0], arr[1]


def conditional_entropy(x, y, nan_strategy=_REPLACE, nan_replace_value=_DEFAULT_REPLACE_VALUE,
                        log_base: float = math.e):
    """
    Calculates the conditional entropy of x given y: S(x|y)
    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
    Parameters:
    -----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of measurements
    nan_strategy : string, default = 'replace'
        How to handle missing values: can be either 'drop' to remove samples
        with missing values, or 'replace' to replace all missing values with
        the nan_replace_value. Missing values are None and np.nan.
    nan_replace_value : any, default = 0.0
        The value used to replace missing values with. Only applicable when
        nan_strategy is set to 'replace'.
    log_base: float, default = e
        specifying base for calculating entropy. Default is base e.
    Returns:
    --------
    float
    """
    if nan_strategy == _REPLACE:
        x, y = replace_nan_with_value(x, y, nan_replace_value)
    elif nan_strategy == _DROP:
        x, y = remove_incomplete_samples(x, y)
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy, log_base)
    return entropy


def theils_u(x, y):
    s_xy = conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


class PPD:

    def __init__(self):
        self.fileName = None
        self.fileExtension = None
        self.data = None
        self.featuresName = None
        self.target_data = None
        self.target_name = None
        self.feature_statistic = None
        self.dataSize = None
        self.dataShape = None
        self.category_list = None
        self.featuresType = None
        self.numericAttrs = None
        self.missingData = None
        self.feature_status = None
        self.simple_imp_data = None
        self.iterative_imp_data = None

        # Outliers variables
        self.Q1 = None
        self.Q3 = None
        self.IQR = None
        self.min_IQR = None
        self.max_IQR = None

    def init(self, file_location):

        client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
        bucket_name = 'django-capstone'
        object_key = file_location
        obj = client.get_object(Bucket=bucket_name, Key=object_key)
        body = obj['Body']
        body_data = body.read().decode('utf-8')

        # Read file using pandas into Data Frame
        # data = pd.read_csv(file_location)
        self.fileName, self.fileExtension = os.path.splitext(file_location)

        if self.fileExtension == ".csv":
            self.data = pd.read_csv(StringIO(body_data))
        elif self.fileExtension == ".xlsx" or self.fileExtension == ".xls":
            self.data = pd.read_excel(file_location)
        elif self.fileExtension == ".json":
            self.data = pd.read_json(file_location)

        # get list of columns name (all features name)
        self.featuresName = self.data.columns

        # Target class data
        self.target_data = self.data.iloc[:, -1]

        # Target Name
        self.target_name = self.featuresName[len(self.featuresName) - 1]

        # create data frame for each feature
        # if it is category type, leave blank
        self.feature_statistic = pd.DataFrame(columns={'Mean', 'Median', 'Variance', 'Standard Deviation'},
                                              index=range(self.featuresName.shape[0]))
        self.feature_statistic.index = self.featuresName

        # size of the data set
        self.dataSize = self.data.size

        # shape of the data set
        self.dataShape = self.data.shape

        # Replace missing value with NAN
        self.replace_missing_by_nan()

        # A list contains all category features name
        self.category_list = list()

        # features datatype
        # first row is attributes type (numeric and categorical)
        # second row is data type of a attribute
        self.featuresType = pd.DataFrame(columns=range(2), index=range(self.featuresName.shape[0]))
        self.featuresType.index = self.featuresName
        self.featuresType.rename(columns={0: 'attr_type', 1: 'n_unique_val'}, inplace=True)

        # initial value for the "featuresType"
        for i in self.featuresType.index:
            self.featuresType.at[i, 'attr_type'] = "null"
            self.featuresType.at[i, 'n_unique_val'] = "null"

        # Get columns name having numeric data
        self.numericAttrs = self.data.select_dtypes(include=np.number, exclude='object').columns

        for i in self.numericAttrs:
            self.featuresType.at[i, 'attr_type'] = 'numeric'

        for i in self.featuresName:
            self.featuresType.at[i, 'n_unique_val'] = self.data.loc[:, i].nunique(dropna=True)
            if self.featuresType.at[i, 'attr_type'] == "null":
                self.featuresType.at[i, 'attr_type'] = 'category'
                self.category_list.append(i)

        # Replace special characters used to represent missing value into NaN
        self.replace_missing_by_nan()

        # Get missing values status
        self.missingData = pd.DataFrame(index=range(self.dataShape[1]), columns=range(2))
        self.missingData.columns = ['number_of_missing', 'percentage_of_missing']
        self.missingData.index = self.featuresName
        self.missingData.loc[:, 'number_of_missing'] = self.data.isnull().sum()
        for i in self.featuresName:
            self.missingData.at[i, 'percentage_of_missing'] = round(
                (self.missingData.at[i, 'number_of_missing'] * 100) / self.dataShape[0], 2)

        self.feature_status = pd.concat([self.featuresType, self.missingData], axis=1)

        # A data frame contains categorical feature name

        # Get statistic of each feature that is numeric type
        self.cal_statistic()

        # Simple Imputer data result variable
        self.simple_imp_data = pd.DataFrame()

        # Iterative Imputer data result variable
        self.iterative_imp_data = pd.DataFrame()

        # Replace NaN with -11111
        # self.data.fillna(value=-11111, inplace=True)

    # Get quartile 1
    def get_Q1(self):
        self.Q1 = self.get_numeric_data().quantile(0.25)
        return self.Q1

    # Get quartile 3
    def get_Q3(self):
        self.Q3 = self.get_numeric_data().quantile(0.75)
        return self.Q3

    # Get IQR
    def get_IQR(self):
        self.IQR = self.Q3 - self.Q1
        return self.IQR

    # Get min IQR
    def get_min_IQR(self):
        self.min_IQR = self.Q1 - 1.5 * self.IQR
        return self.min_IQR

    # Get max IQR
    def get_max_IQR(self):
        self.max_IQR = self.Q3 + 1.5 * self.IQR
        return self.max_IQR

    # Calculate Quartile
    def cal_quartile(self):
        temp = self.get_numeric_data()
        self.get_missing_status()
        numeric_missing_feature = self.missingData.T.filter(temp.columns).columns
        if len(numeric_missing_feature) > 0:
            temp = temp.loc[temp[numeric_missing_feature[0]].notnull(), :]
        self.Q1 = temp.quantile(0.25)
        self.Q3 = temp.quantile(0.75)
        self.IQR = self.Q3 - self.Q1
        self.min_IQR = self.Q1 - 1.5 * self.IQR
        self.max_IQR = self.Q3 + 1.5 * self.IQR

    # Get a feature outlier data
    def get_feature_Outlier_data(self, feature_name):
        numeric_data = pd.DataFrame(self.get_numeric_data()[feature_name])
        numeric_data = pd.DataFrame(numeric_data.loc[numeric_data[feature_name].notnull(), feature_name])
        Q1 = numeric_data.quantile(0.25)
        Q3 = numeric_data.quantile(0.75)
        IQR = Q3 - Q1
        min_IQR = Q1 - 1.5 * IQR
        max_IQR = Q3 + 1.5 * IQR
        # numeric_data = numeric_data.notna()
        outlier_data = numeric_data[
            (numeric_data[feature_name] < min_IQR[0]) |
            (numeric_data[feature_name] > max_IQR[0])
            ][feature_name]
        return outlier_data

    # get a feature number of outlier
    def get_feature_num_outlier(self, feature_name):
        return self.get_feature_Outlier_data(feature_name).shape[0]

    # Remove a feature outliers by turning them into missing value
    def remove_feature_outlier_data(self, feature_name):
        outlier_data = self.get_feature_Outlier_data(feature_name)
        self.data.loc[self.data.index[outlier_data.index], feature_name] = np.NaN

    # Check if a feature has missing data
    def check_feature_missing(self, feature_name):
        self.get_missing_status()
        if self.missingData.loc[feature_name, 'number_of_missing'] > 0:
            return True
        return False

    # Return full data
    def get_data(self):
        return self.data

    # Get data of 2 features
    def get_features_data(self, feature_list):
        return self.data.loc[:, feature_list]

    # Transform all types of symbol that represent missing data into NAN
    def replace_missing_by_nan(self):
        self.data.replace(r'^\s*$', np.nan, regex=True)
        for row in self.data.iterrows():
            for column_index in row[1].index:
                var = self.data.at[row[0], column_index]
                if var == '-' or var == '/':
                    self.data.at[row[0], column_index] = np.NaN

    def get_numeric_features_name(self):
        temp = self.data.select_dtypes(include='number', exclude='object').columns
        return temp

    def get_numeric_data(self):
        temp = self.data.select_dtypes(include='number', exclude='object')
        return temp

    def getFeatureName(self):
        # get list of columns name
        self.featuresName = self.data.columns
        return self.featuresName

    def getDataSize(self):
        # size of the data set
        self.dataSize = self.data.size
        return self.dataSize

    def getDataShape(self):
        # shape of the data set
        self.dataShape = self.data.shape
        return self.dataShape

    def remove_feature(self, feature):
        self.data = self.data.drop(feature, axis=1)
        self.featuresName = self.data.columns
        self.dataShape = self.data.shape

    def getFeatureType(self):
        # features datatype
        # first row is attributes type (numeric and categorical)
        # second row is data type of a attribute
        self.featuresName = None
        self.featuresType = None
        self.category_list = list()
        self.featuresName = self.data.columns
        self.featuresType = pd.DataFrame(columns=range(2), index=range(self.featuresName.shape[0]))
        self.featuresType.index = self.featuresName
        self.featuresType.rename(columns={0: 'attr_type', 1: 'n_unique_val'}, inplace=True)

        # initial value for the "featuresType"
        for i in self.featuresType.index:
            self.featuresType.at[i, 'attr_type'] = "null"
            self.featuresType.at[i, 'n_unique_val'] = "null"

        # Get columns name having numeric data
        self.numericAttrs = self.data.select_dtypes(include=np.number, exclude='object').columns

        for i in self.numericAttrs:
            self.featuresType.at[i, 'attr_type'] = 'numeric'

        for i in self.featuresName:
            self.featuresType.at[i, 'n_unique_val'] = self.data.loc[:, i].nunique(dropna=True)
            if self.featuresType.at[i, 'attr_type'] == "null":
                self.featuresType.at[i, 'attr_type'] = 'category'
                self.category_list.append(i)

    # Get number of numeric and categorical features
    def get_num_numeric_and_category_feature(self):
        self.getFeatureType()
        return len(self.numericAttrs), len(self.category_list)

    # Get total number of missing values in the data set and the percentage
    def get_num_missing(self):
        self.get_missing_status()
        temp = 0
        for i in self.missingData.index:
            temp = temp + self.missingData.loc[i, 'number_of_missing']

        percentage = round((temp * 100) / self.dataShape[0], 2)
        return temp, percentage

    def get_category_list(self):
        self.getFeatureType()
        cat_list = pd.DataFrame(columns={'feature', 'n_unique_val'})
        for i in self.category_list:
            cat_list.at[i, 'feature'] = i
            cat_list.at[i, 'n_unique_val'] = self.featuresType.loc[i, 'n_unique_val']
        return cat_list

    def get_missing_status(self):
        self.missingData = pd.DataFrame(index=range(self.dataShape[1]), columns=range(2))
        self.missingData.columns = ['number_of_missing', 'percentage_of_missing']
        self.missingData.index = self.featuresName
        self.missingData.loc[:, 'number_of_missing'] = self.data.isnull().sum()
        # for i in self.featuresName:
        #     self.missingData.at[i, 'number_of_missing'] = self.data[self.data[i] == np.NAN].shape[0]
        for i in self.featuresName:
            self.missingData.at[i, 'percentage_of_missing'] = round(
                (self.missingData.at[i, 'number_of_missing'] * 100) / self.dataShape[0], 2)

    def get_missing_data(self):
        self.get_missing_status()
        return self.missingData

    def check_any_missing(self):
        self.get_missing_status()
        for i in self.missingData.index:
            if self.missingData.at[i, 'number_of_missing'] > 0:
                return True
        return False

    def get_feature_status(self):
        self.getFeatureType()
        self.cal_statistic()
        self.get_missing_status()
        self.feature_status = pd.concat([self.featuresType, self.missingData, self.feature_statistic], axis=1)
        return self.feature_status

    def cal_statistic(self):
        for i in self.featuresName:
            if self.featuresType.at[i, 'attr_type'] == 'numeric':
                temp = pd.DataFrame(self.data.loc[:, i])
                self.feature_statistic.at[i, 'Mean'] = round(temp.mean(skipna=True)[0], 2)
                self.feature_statistic.at[i, 'Median'] = round(temp.median(skipna=True)[0], 2)
                self.feature_statistic.at[i, 'Variance'] = round(temp.var(skipna=True)[0], 2)
                self.feature_statistic.at[i, 'Standard Deviation'] = round(temp.std(skipna=True)[0], 2)
            else:
                self.feature_statistic.at[i, 'Mean'] = None
                self.feature_statistic.at[i, 'Median'] = None
                self.feature_statistic.at[i, 'Variance'] = None
                self.feature_statistic.at[i, 'Standard Deviation'] = None

    def update_data(self):
        self.featuresName = self.data.columns
        self.dataShape = self.data.shape

    # This label encoding function is for all categorical feature
    def auto_label_encoding(self, features):
        le = preprocessing.LabelEncoder()
        if features == 'all':
            self.getFeatureType()
            for i in self.category_list:
                self.data.loc[self.data[i].notnull(), i] = le.fit_transform(self.data.loc[self.data[i].notnull(), i])
        else:
            self.data.loc[self.data[features].notnull(), features] = le.fit_transform(self.data.loc[self.data[features].notnull(), features])

    def one_hot_encode(self, feature, drop, sparse, dType, handle_unknown):
        ohe = preprocessing.OneHotEncoder(
            drop=drop,
            sparse=sparse,
            # dtype=dType,
            handle_unknown=handle_unknown
        )
        if self.check_any_missing() is False:
            if feature == 'all':
                self.data = pd.get_dummies(self.data.iloc[:, :-1])
                self.data = pd.concat([self.data, self.target_data], axis=1)
            else:
                temp = pd.get_dummies(self.data.loc[:, feature])
                self.data = self.data.drop(feature, axis=1)
                self.data = pd.concat([self.data, temp], axis=1)
                target = self.data.pop(self.target_name)
                self.data = pd.concat([self.data, target], axis=1)

            self.update_data()

    def binary_encode(self, feature, verbose, drop_invariant, return_df, handle_unknown, handle_missing):
        be = ce.BinaryEncoder(
            verbose=verbose,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing
        )
        if self.check_any_missing() is False:
            if feature == 'all':
                self.data = be.fit_transform(self.data.iloc[:, :-1])
                self.data = pd.concat([self.data, self.target_data], axis=1)
            else:
                temp = be.fit_transform(self.data.loc[:, feature])
                self.data = self.data.drop(feature, axis=1)
                self.data = pd.concat([self.data, temp], axis=1)
                target = self.data.pop(self.target_name)
                self.data = pd.concat([self.data, target], axis=1)

            self.update_data()

    def hashing_encoder(self, feature, verbose, drop_invariant, return_df, hash_method, max_process, max_sample):
        he = ce.HashingEncoder(
            verbose=verbose,
            drop_invariant=drop_invariant,
            return_df=return_df,
            hash_method=hash_method,
            max_process=max_process,
            max_sample=max_sample
        )
        if self.check_any_missing() is False:
            if feature == 'all':
                self.data = he.fit_transform(self.data.iloc[:, :-1])
                self.data = pd.concat([self.data, self.target_data], axis=1)
            else:
                temp = he.fit_transform(self.data.loc[:, feature])
                self.data = self.data.drop(feature, axis=1)
                self.data = pd.concat([self.data, temp], axis=1)
                target = self.data.pop(self.target_name)
                self.data = pd.concat([self.data, target], axis=1)

            self.update_data()

    def sum_encode(self, feature, verbose, drop_invariant, return_df, handle_unknown, handle_missing):
        se = ce.SumEncoder(
            verbose=verbose,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing
        )
        if self.check_any_missing() is False:
            if feature == 'all':
                self.data = se.fit_transform(self.data.iloc[:, :-1])
                self.data = pd.concat([self.data, self.target_data], axis=1)
            else:
                temp = se.fit_transform(self.data.loc[:, feature])
                self.data = self.data.drop(feature, axis=1)
                self.data = pd.concat([self.data, temp], axis=1)
                target = self.data.pop(self.target_name)
                self.data = pd.concat([self.data, target], axis=1)

            self.update_data()

    def target_encoder(self, feature, verbose, drop_invariant, return_df, handle_unknown, min_sample_leaf, smoothing):
        le = preprocessing.LabelEncoder()
        self.data.loc[self.data[self.target_name].notnull(), self.target_name] = le.fit_transform(
            self.data.loc[self.data[self.target_name].notnull(), self.target_name]
        )
        te = ce.TargetEncoder(
            verbose=verbose,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            min_samples_leaf=min_sample_leaf,
            smoothing=smoothing
        )
        if feature == 'all':
            self.getFeatureType()
            for i in self.category_list:
                if i != self.target_name:
                    temp = self.data.loc[self.data[i].notnull(), i].index
                    self.data.loc[self.data[i].notnull(), i] = \
                        te.fit_transform(
                        self.data.loc[self.data[i].notnull(), i],
                        self.data.loc[temp, self.target_name]
                    )
        else:
            self.data.loc[self.data[feature].notnull(), feature] = te.fit_transform(
                self.data.loc[self.data[feature].notnull(), feature],
                self.data.loc[self.data[self.target_name].notnull(), self.target_name]
            )

    def leave_one_out_encode(self, feature, verbose, drop_invariant, return_df, handle_unknown, handle_missing, sigma):
        le = preprocessing.LabelEncoder()
        self.data.loc[self.data[self.target_name].notnull(), self.target_name] = le.fit_transform(
            self.data.loc[self.data[self.target_name].notnull(), self.target_name]
        )
        loue = ce.LeaveOneOutEncoder(
            verbose=verbose,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
            sigma=sigma
        )
        if feature == 'all':
            self.getFeatureType()
            for i in self.category_list:
                if i != self.target_name:
                    temp = self.data.loc[self.data[i].notnull(), i].index
                    self.data.loc[self.data[i].notnull(), i] = \
                        loue.fit_transform(
                        self.data.loc[self.data[i].notnull(), i],
                        self.data.loc[temp, self.target_name]
                    )
        else:
            self.data.loc[self.data[feature].notnull(), feature] = loue.fit_transform(
                self.data.loc[self.data[feature].notnull(), feature],
                self.data.loc[self.data[self.target_name].notnull(), self.target_name]
            )

    def simple_imputer(self, feature_type, strategy, fill_value, verbose):
        self.getFeatureType()
        imp = SimpleImputer(
            missing_values=np.NAN,
            strategy=strategy,
            fill_value=fill_value,
            verbose=verbose,
            # copy=copy,
            # add_indicator=add_indicator
        )
        if feature_type == 'all':
            self.data = pd.DataFrame(imp.fit_transform(self.data))
        if feature_type == 'numeric_features':
            # transformer = ColumnTransformer(transformers=[('num', imp, self.numericAttrs)])
            # self.data[[self.numericAttrs]] = transformer.fit_transform(self.data[[self.numericAttrs]])
            self.data.loc[:, self.numericAttrs] = imp.fit_transform(self.data.loc[:, self.numericAttrs])
        if feature_type == 'categorical_features':
            self.data.loc[:, self.category_list] = imp.fit_transform(self.data.loc[:, self.category_list])
        self.data.columns = self.featuresName
        self.data = self.data.infer_objects()

    def iterative_imputer(self, estimator, max_iter, tol, n_nearest_feature, initial_strategy,
                          imputation_order, skip_complete, min_value, max_value, verbose, random_state):
        print("Interative Imputer")
        print(n_nearest_feature)
        my_estimator = None

        if estimator == 'BayesianRidge':
            my_estimator = BayesianRidge()
        if estimator == 'DecisionTreeRegressor':
            my_estimator = DecisionTreeRegressor()
        if estimator == 'ExtraTreesRegressor':
            my_estimator = ExtraTreesRegressor()
        if estimator == 'KNeighborsRegressor':
            my_estimator = KNeighborsRegressor()
        if estimator == 'DecisionTreeClassifier':
            my_estimator = DecisionTreeClassifier

        imp = IterativeImputer(
            estimator=my_estimator,
            missing_values=np.NAN,
            # sample_posterior=sample_posterior,
            max_iter=max_iter,
            tol=tol,
            n_nearest_features=n_nearest_feature,
            initial_strategy=initial_strategy,
            imputation_order=imputation_order,
            skip_complete=skip_complete,
            min_value=min_value,
            max_value=max_value,
            verbose=verbose,
            random_state=random_state,
            # add_indicator=add_indicator
        )

        print("Iterative Imputer is created")
        self.data = imp.fit_transform(self.data)
        self.data = pd.DataFrame(self.data)
        self.data.columns = self.featuresName
        self.data = self.data.infer_objects()

    # def outlier_detection_result(self):
    #     random_state = np.random.RandomState(42)
    #     # detect 5% observations that are not similar to the rest of the data
    #     outliers_fraction = 0.05
    #     # Define seven outlier detection tools to be compared
    #     classifiers = {
    #         'Angle-based Outlier Detector (ABOD)': ABOD(contamination=outliers_fraction),
    #         'Cluster-based Local Outlier Factor (CBLOF)': CBLOF(
    #             contamination=outliers_fraction,
    #             check_estimator=False,
    #             random_state=random_state
    #         ),
    #         'Feature Bagging': FeatureBagging(
    #             LOF(n_neighbors=35),
    #             contamination=outliers_fraction,
    #             check_estimator=False,
    #             random_state=random_state
    #         ),
    #         'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=outliers_fraction),
    #         'Isolation Forest': IForest(
    #             contamination=outliers_fraction,
    #             random_state=random_state
    #         ),
    #         'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
    #         'Average KNN': KNN(
    #             method='mean',
    #             contamination=outliers_fraction
    #         )
    #     }

    def get_corr_matrix(self):
        self.update_data()
        self.data = self.data.infer_objects()
        self.getFeatureType()
        corr = pd.DataFrame(index=self.data.columns, columns=self.data.columns)

        for i in corr.index:
            for j in corr.columns:
                if i in self.numericAttrs:
                    if j in self.numericAttrs:
                        corr.at[i, j] = round(np.corrcoef(self.data.loc[:, i], self.data.loc[:, j])[0, 1], 2)
                    else:
                        corr.at[i, j] = round(correlation_ratio(self.data.loc[:, j], self.data.loc[:, i]), 2)
                else:
                    if j in self.numericAttrs:
                        corr.at[i, j] = round(correlation_ratio(self.data.loc[:, i], self.data.loc[:, j]), 2)
                    else:
                        corr.at[i, j] = round(theils_u(self.data.loc[:, i], self.data.loc[:, j]), 2)

        return corr

    def set_standardization(self, feature):
        num_feature = self.data.select_dtypes(include=np.number, exclude='object').columns
        if (feature is None) | (feature == '') | (feature == 'all'):
            self.data.loc[:, num_feature] = preprocessing.scale(self.data[num_feature])
        else:
            self.data.loc[:, feature] = preprocessing.scale(self.data[[feature]])

    def set_normalization(self, feature):
        num_feature = self.data.select_dtypes(include=np.number, exclude='object').columns
        if (feature is None) | (feature == '') | (feature == 'all'):
            self.data.loc[:, num_feature] = preprocessing.normalize(self.data[num_feature])
        else:
            self.data.loc[:, feature] = preprocessing.normalize(self.data[[feature]])

    def get_category_stat(self, feature):
        temp = pd.DataFrame(self.data[feature].value_counts())
        temp.columns = [''] * len(temp.columns)
        temp = temp.to_dict()
        temp = temp[''].items()
        return temp
