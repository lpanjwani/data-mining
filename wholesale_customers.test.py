# import library
import unittest
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Import Test File
import wholesale_customers

# create a class
class TestWholesaleCustomers(unittest.TestCase):
    def setUp(self):
      self.csv = wholesale_customers.read_csv_2('data/wholesale_customers.csv')

    def test_read_csv_2(self):
      expected = pd.read_csv('data/wholesale_customers.csv').drop(columns=['Channel', 'Region'])

      self.assertEqual(self.csv.to_json(), expected.to_json())

    def test_summary_statistics(self):
      expected = self.csv.describe().transpose()
      received = wholesale_customers.summary_statistics(self.csv)

      self.assertEqual(received.to_json(), expected.to_json())

    def test_standardize(self):
      expected = (self.csv - self.csv.mean()) / self.csv.std()
      received = wholesale_customers.standardize(self.csv)

      self.assertEqual(received.to_json(), expected.to_json())

    def test_kmeans(self):
      expected = KMeans(n_clusters=2).fit(self.csv).labels_
      received = wholesale_customers.kmeans(self.csv, 2)

      self.assertEqual(received.all(), expected.all())

    def test_kmeans_plus(self):
      expected = KMeans(n_clusters=2, init='k-means++').fit(self.csv).labels_
      received = wholesale_customers.kmeans_plus(self.csv, 2)

      self.assertEqual(received.all(), expected.all())

    def test_agglomerative(self):
      expected = AgglomerativeClustering(n_clusters=2).fit(self.csv).labels_
      received = wholesale_customers.agglomerative(self.csv, 2)

      self.assertEqual(received.all(), expected.all())

    def test_clustering_score(self):
      labels = wholesale_customers.kmeans(self.csv, 2)

      expected = silhouette_score(self.csv, labels)
      received = wholesale_customers.clustering_score(self.csv, labels)

      self.assertEqual(received, expected)

    # TODO: Create this test after verification
    def test_cluster_evaluation(self):
        received = wholesale_customers.cluster_evaluation(self.csv)

        # self.assertEqual(received.to_json(), expected.to_json())

    def best_clustering_score(self):
        expected = 0.542401
        received = wholesale_customers.best_clustering_score(wholesale_customers.cluster_evaluation(self.csv))

        self.assertEqual(received, expected)


# driver code
if __name__ == '__main__':
    unittest.main()