import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GroupKFold
from sklearn.base import clone

# Reads data from csv file
# Input: filename
# Output: dataframe
# Indexes by first column, which is student ID
# Each row is a student, each column is a feature
def readData(filename):
    df = pd.read_csv(filename,index_col=0)
    return df

# Replaces outliers greater than 99th percentile in specified column
# Input: dataframe, column name
# Output: dataframe
def replaceColumnOutliers(df, columnName):
    data = getColumnValues(df, columnName)
    cutoff = findPercentile(data, 99)
    df.loc[df[columnName] > cutoff, columnName] = cutoff
    return df

# Gets a copy of specified column
# Input: dataframe, column name
# Output: values of column
def getColumnValues(df, columnName):
    data = df[columnName].copy().values
    return data

# Finds specified percentile of a given feature
# Input: column of values, percentile
# Output: value of percentile
def findPercentile(data, perc):
    cutoff = np.percentile(data, perc)
    return cutoff

# Replaces outliers in a list of specified columns
# Input: dataframe, list of column names
# Output: dataframe
def replaceOutliers(df, columnNames):
    for c in columnNames:
        df = replaceColumnOutliers(df,c)
    return df

# Scales a single column to [0,1]
# Input: dataframe, column name
# Output: dataframe, scaler
def scaleColumn(df, columnName):
    data = getColumnValues(df, columnName)
    data = data.reshape(-1,1)
    scaler = MinMaxScaler()
    df[columnName] = np.reshape(scaler.fit_transform(data),len(data))
    return df, scaler

# Scales all columns to [0,1]
# Input: dataframe
# Output: scaled dataframe, list of scalers for each column
def scaleColumns(df):
    scalers = []
    newdf = df.copy()
    for c in getColumnLabels(newdf):
        newdf, scaler = scaleColumn(newdf, c)
        scalers.append(scaler)
    return newdf, scalers

# Get list of dataframe features
# Input: dataframe
# Output: list of text labels
def getColumnLabels(df):
    return list(df)

# Drops specified columns from a dataframe
# Input: dataframe, list of column names
# Output: dataframe
def dropColumns(df, columnNames):
    return df.drop(columns=columnNames)

# Filters dataframe using specified index
# Input: dataframe to filter, index to filter by
# Output: filtered dataframe
def filterByIndex(df, ind):
    df = df[df.index.isin(ind)]
    return df

# Gets a specified sample of rows
# Input: dataframe, number of rows
# Output: sampled dataframe
def getSample(df, num):
    return df.sample(n=num)

# Trains a cluster algorithm
# Input: algorithm object, dataframe
# Output: series of cluster labels, model
def cluster(algorithm, df):
    data = df.values
    model = algorithm.fit(data)
    clusterLabels = model.predict(data)
    labelSeries = pd.Series(clusterLabels,index=df.index)
    return labelSeries, model

# Displays histogram of cluster distribution and centers of clusters
# Input: array of cluster labels, array of cluster centers
# Output: display histogram, dataframe with centers and proportions
def displayClusters(labels, clusterReps,columns):
    clusterDict = countLabels(labels)
    factor = 1.0 / sum(clusterDict.values())
    for k in clusterDict:
        clusterDict[k] = clusterDict[k] * factor
    plt.bar(list(clusterDict.keys()), clusterDict.values())
    proportions = pd.DataFrame.from_dict(clusterDict, orient='index', columns=['Proportions']).round(2)
    centers = pd.DataFrame(clusterReps, columns=columns).round(2)
    return centers.join(proportions)

# Counts number of items in each cluster
# Input: array of cluster labels
# Output: dictionary of counts, keys are cluster numbers
def countLabels(labels):
    clusters, counts = np.unique(labels, return_counts=True)
    return dict(zip(clusters, counts))

# Inverts scaling function to original values
# Input: dataframe, array of scalers
# Output: dataframe
def reverseScale(df, scalers):
    for i in range(len(scalers)):
        data = df.iloc[:,[i]].values
        unscaled = scalers[i].inverse_transform(data)
        df.iloc[:,[i]] = unscaled
    return df

# Trains Bayesian Ridge Regression Model
# Input: training features, training labels, test features
# Output: array of predictions, trained model
def classify(X_train, y_train, X_test):
    model = clone(BayesianRidge()).fit(X_train, y_train)
    return model.predict(X_test), model

# Calculates Spearman correlation between labels and predictions
# Input: test labels, predictions as vectors
# Output: Spearman correlation
def evaluateSpearman(y_test, predictions):
    rho, _ = spearmanr(y_test, predictions)
    return rho

# Calculates MSE between labels and predictions
# Input: test labels, predictions as vectors
# Output: mean squared error
def evaluateMSE(y_test, predictions):
    return mean_squared_error(y_test,predictions)