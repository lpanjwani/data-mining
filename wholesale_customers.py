import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Part 2: Cluster Analysis

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
	return pd.read_csv(data_file).drop(columns=['Channel', 'Region'], axis=1)

# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns.
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
	return df.describe().transpose()

# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
	df_copy = df.copy()
	return (df_copy - df_copy.mean()) / df_copy.std()

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
def kmeans(df, k):
	return pd.Series(KMeans(n_clusters=k, n_init=10).fit(df).labels_)

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
	return pd.Series(KMeans(n_clusters=k, init='k-means++').fit(df).labels_)

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
	return pd.Series(AgglomerativeClustering(n_clusters=k).fit(df).labels_)

# Given a data set X and an assignment to clusters y
# return the Solhouette score of the clustering.
def clustering_score(X,y):
	return silhouette_score(X, y)

# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the:
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative',
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
	nIterations = 10
	kValues = [3, 5, 10]

	rdf = pd.DataFrame({
		'Algorithm': [],
		'data': [],
		'k': [],
		'Silhouette Score': []
	})

	for kValue in kValues:
		AgglomerativeOriginal = agglomerative(df, kValue)
		rdf = rdf.append({
			'Algorithm': "Agglomerative",
			'data': "Original",
			'k': kValue,
			'Silhouette Score': clustering_score(df, AgglomerativeOriginal)
        }, ignore_index=True)

		AgglomerativeStandardized = agglomerative(standardize(df), kValue)
		rdf = rdf.append({
			'Algorithm': "Agglomerative",
			'data': "Standardized",
			'k': kValue,
			'Silhouette Score': clustering_score(standardize(df), AgglomerativeStandardized)
        }, ignore_index=True)

		for n in range(nIterations):
			kMeansOriginal = kmeans(df, kValue)
			rdf = rdf.append({
				'Algorithm': "Kmeans",
				'data': "Original",
				'k': kValue,
				'Silhouette Score': clustering_score(df, kMeansOriginal)
			}, ignore_index=True)

			kMeansStandardized = kmeans(standardize(df), kValue)
			rdf = rdf.append({
				'Algorithm': "Kmeans",
				'data': "Standardized",
				'k': kValue,
				'Silhouette Score': clustering_score(standardize(df), kMeansStandardized)
			}, ignore_index=True)


	return rdf

# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
	return max(rdf['Silhouette Score'])

# Run some clustering algorithm of your choice with k=3 and generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):
	kMeans = kmeans(df, 3)
	df['Cluster'] = kMeans

	fig, axes = plt.subplots(nrows=len(df.columns), ncols=len(df.columns), figsize=(50, 50))

	for x_index, x in enumerate(df.columns):
		for y_index, y in enumerate(df.columns):
			if x == y:
				continue

			df.plot.scatter(x=x, y=y, c=df['Cluster'], colormap='viridis', ax=axes[x_index, y_index], title=x + " vs " + y, xlabel=x, ylabel=y)

	fig.show()
