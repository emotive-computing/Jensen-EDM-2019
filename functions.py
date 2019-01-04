# Data Manipulation
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

# Visualization
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
#import graphviz
import matplotlib.cm as cm
from sklearn import tree

# Calculations
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_samples, silhouette_score

# Machine Learning
from sklearn.cluster import KMeans, Birch
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import KFold, GroupKFold
from sklearn.linear_model import BayesianRidge
from sklearn.base import clone
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaders
from sklearn.externals import joblib


#########################################################
### ---------- DATA MANIPULATION FUNCTIONS ---------- ###
#########################################################

# Reads data from csv file
# Input: filename
# Output: dataframe
# Indexes by first column, which is student ID
# Each row is a student, each column is a feature
def readData(filename):
    df = pd.read_csv(filename,index_col=0)
    return df

# Filters dataframe using specified index
# Input: dataframe to filter, index to filter by
# Output: filtered dataframe
def filterByIndex(df, ind):
    df = df[df.index.isin(ind)]
    return df

# Keeps students who answered survey for specified state
# Input: string of state, dataframe of survey data
# Output: dataframe of survey features, series of labels
def filterByState(state, df):
    df = df[df['survey_question'] == state]
    labels = df['survey_answer']
    features = df.drop(columns=['survey_answer','survey_question'])
    return features, labels

# Gets a copy of specified column
# Input: dataframe, column name
# Output: values of column
def getColumnValues(df, columnName):
    data = df[columnName].copy().values
    return data

# Drops specified columns from a dataframe
# Input: dataframe, list of column names
# Output: dataframe
def dropColumns(df, columnNames):
    return df.drop(columns=columnNames)

# Replaces outliers greater than 99th percentile in specified column
# Input: dataframe, column name
# Output: dataframe
def replaceColumnOutliers(df, columnName):
    data = getColumnValues(df, columnName)
    cutoff = findPercentile(data, 99)
    df.loc[df[columnName] > cutoff, columnName] = cutoff
    return df

# Replaces outliers in a list of specified columns
# Input: dataframe, list of column names
# Output: dataframe
def replaceDataFrameOutliers(df, columnNames):
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
def scaleDataframe(df):
    scalers = []
    newdf = df.copy()
    for c in getColumnLabels(newdf):
        newdf, scaler = scaleColumn(newdf, c)
        scalers.append(scaler)
    return newdf, scalers

# Inverts scaling function to original values
# Input: dataframe, array of scalers
# Output: dataframe
def reverseScale(df, scalers):
    for i in range(len(scalers)):
        data = df.iloc[:,[i]].values
        unscaled = scalers[i].inverse_transform(data)
        df.iloc[:,[i]] = unscaled
    return df

# Gets a specified sample of rows
# Input: dataframe, number of rows
# Output: sampled dataframe
def getSample(df, num):
    return df.sample(n=num)

# Get list of dataframe features
# Input: dataframe
# Output: list of text labels
def getColumnLabels(df):
    return list(df)

# Counts number of items in each cluster
# Input: array of cluster labels
# Output: dictionary of counts, keys are cluster numbers
def countLabels(labels):
    clusters, counts = np.unique(labels, return_counts=True)
    return dict(zip(clusters, counts))

# Splits features by cluster
# Input: dataframe of features, series of labels, series of cluster labels
# Output: feature dictionary, label dictionary, group dictionary
# dictionaries have cluster number as key
def splitClusters(featuredf, labelSeries, clusterSeries):
    uniqueLabels = np.unique(clusterSeries.values)
    clusterDict = {}
    labelDict = {}
    groupDict = {}
    for number in uniqueLabels:
        # Keep survey data that has been clustered
        labelsInCluster = filterByIndex(labelSeries,clusterSeries.index)
        labels = labelsInCluster.values
        featuresInCluster = filterByIndex(featuredf, clusterSeries.index)
        features = featuresInCluster.values
        groups = featuresInCluster.index.values
        # Get cluster labels for each survey response
        clusterLabels = np.array([clusterSeries[i] for i in labelsInCluster.index])
        # filter by observations in cluster
        cluster = features[clusterLabels == number,:]
        clusterDict[number] = cluster
        label = labels[clusterLabels == number]
        labelDict[number] = label
        group = groups[clusterLabels == number]
        groupDict[number] = group
    return clusterDict, labelDict, groupDict

# Creates dataframe from a dictionary
# Input: dictionary
# Output: dataframe
# Rows are the keys of the dictionary 
def dataframeFromDict(d):
    return pd.DataFrame.from_dict(d,orient='index')


###################################################
### ---------- CALCULATION FUNCTIONS ---------- ###
###################################################

# Finds specified percentile of a given feature
# Input: column of values, percentile
# Output: value of percentile
def findPercentile(data, perc):
    cutoff = np.percentile(data, perc)
    return cutoff

# Calculates pairwise distances between samples
# Input: dataframe
# Output: summary statistics and histogram of distances
def calculateDistances(df):
    distances = pairwise_distances(df.values)
    distances = distances[np.triu_indices(1000,k=1)]
    dist_df = pd.DataFrame(distances)
    dist_df.columns = ['Distance']
    return dist_df

# Calculates silhouette score for specified range of clusters
# Input: array of cluster values, array of values to cluster
# Output: prints silhouette score for each value of clusters
def silhouetteSummary(clusters, data):
    for n_clusters in clusters:
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, labels)
        print('For',n_clusters,'clusters, average silhouette score is',silhouette_avg)
        
# Calculates Spearman correlation between labels and predictions
# Input: test labels, predictions
# Output: Spearman correlation
def evaluateSpearman(y_test, predictions):
    rho, _ = spearmanr(y_test, predictions)
    return rho

# Calculates number of responses and mean responses per cluster per state
# Input: list of states, number of clusters, dataframe of survey features, cluster series
# Output: array of counts, array of means
def surveySummary(states, nClusters, surveydf, clusterSeries):
    counts = np.zeros((len(states),nClusters))
    means = np.zeros((len(states),nClusters))
    for s in range(len(states)):
        state = states[s]
        featuredf, surveySeries = filterByState(state,surveydf)
        featureDict, labelDict, groupDict = splitClusters(featuredf,surveySeries,clusterSeries)
        # now have survey responses for all clusters in state s
        for c in labelDict.keys():
            # get survey responses for state s in cluster c
            answers = labelDict[c]

            # count number of responses for state s in cluster c
            count = len(answers)
            counts[s,c] = count

            # get mean survey answer for state s in cluster c
            mean = np.mean(answers)
            means[s,c] = mean
    return counts, means


########################################################
### ---------- MACHINE LEARNING FUNCTIONS ---------- ###
########################################################

# Trains a cluster algorithm
# Input: algorithm object, dataframe
# Output: series of cluster labels, model
def cluster(algorithm, df):
    data = df.values
    model = algorithm.fit(data)
    clusterLabels = model.predict(data)
    labelSeries = pd.Series(clusterLabels,index=df.index)
    return labelSeries, model

# Trains Bayesian Ridge Regression Model
# Input: training features, training labels, test features
# Output: array of predictions, trained model
def classify(X_train, y_train, X_test):
    model = clone(BayesianRidge()).fit(X_train, y_train)
    return model.predict(X_test), model


#####################################################
### ---------- VISUALIZATION FUNCTIONS ---------- ###
#####################################################

# Plots cumulative distribution function for a given feature
# Input: name for a given feature, dataframe
# Output: displays graph
def plotCDF(columnName, df):
    data = getColumnValues(df, columnName)
    data.sort()
    plt.step(data, np.arange(data.size))
    plt.title(columnName)

# Displays the rounded summary statistics of a dataframe
# Input: dataframe
# Output: displays the summary statistics, prints number of samples
def summaryStats(df,scalers=None):
    num = len(df)
    print(num,'samples')
    if scalers is not None:
        df = reverseScale(df, scalers)
    return df.describe().drop(['count']).round(2)

# Displays boxplots of the specified features
# Input: dataframe, list of column names, boolean for vertical
# Output: displays boxplots
def makeBoxplots(df, columns, isVertical):
    df.boxplot(column=columns, figsize=(15,8), grid=False, vert=isVertical)
    
# Displays histogram of each feature of a dataframe
# Input: dataframe
# Output: displays grid of histograms
def makeHistograms(df):
    df.hist(figsize=(16,14))
    
# Displays grid of scatter plots comparing features
# Input: dataframe
# Output: displays grid of scatter plots
def makeScatterMatrix(df):
    scatter_matrix(df, figsize=(30,30))
    
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

# Displays decision tree for given set of clusters
# Input: array of data, cluster labels, target filename, depth of tree, list of column names
# Output: image file of decision tree
def makeTree(data,labels,filename,depth,columns):
    modelTree = tree.DecisionTreeClassifier(max_depth=depth)
    modelTree = modelTree.fit(data,labels)
    dot_data = tree.export_graphviz(modelTree,out_file=None,feature_names=columns,proportion=True,precision=2,filled=True)
    graph = graphviz.Source(dot_data,format='png')
    graph.render(filename=filename,view=True)
    
# Displays colored table of correlation coefficients
# Input: dataframe
# Output: dataframe
def makeCorrelationTable(df):
    table = df.corr(method='spearman').round(2).style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'))
    return table