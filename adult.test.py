# import library
import unittest
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Import Test File
import adult

# create a class
class TestAdult(unittest.TestCase):
    def setUp(self):
      self.csv = adult.read_csv_1('data/adult.csv')

    def test_read_csv_1(self):
      expected = pd.read_csv('data/adult.csv')
      expected = expected.drop("fnlwgt", axis=1)

      self.assertEqual(self.csv.to_json(), expected.to_json())

    def test_num_rows(self):
      expected = self.csv.shape[0]

      received = adult.num_rows(self.csv)

      self.assertEqual(received, expected)

    def test_column_names(self):
      expected = self.csv.columns.values
      received = adult.column_names(self.csv)

      self.assertEqual(received.all(), expected.all())

    def test_missing_values(self):
      expected = self.csv.isnull().values.sum()
      received = adult.missing_values(self.csv)

      self.assertEqual(received, expected)

    def test_columns_with_missing_values(self):
      expected = list(self.csv.columns[self.csv.isnull().any()])
      received = adult.columns_with_missing_values(self.csv)

      self.assertEqual(received, expected)

    def test_bachelors_masters_percentage(self):
      expected = 0.0
      received = adult.bachelors_masters_percentage(self.csv)

      self.assertEqual(received, expected)

    def test_bachelors_masters_percentage(self):
      bachelors = self.csv[self.csv['education'] == 'Bachelors']
      masters = self.csv[self.csv['education'] == 'Masters']
      expected = round((len(bachelors) + len(masters)) / len(self.csv), 3)

      received = adult.bachelors_masters_percentage(self.csv)

      self.assertEqual(received, expected)


    def test_data_frame_without_missing_values(self):
      expected = self.csv.dropna()
      received = adult.data_frame_without_missing_values(self.csv)

      self.assertEqual(received.to_json(), expected.to_json())

    def test_one_hot_encoding(self):
      expected = pd.get_dummies(self.csv.loc[:, self.csv.columns != 'class'])

      received = adult.one_hot_encoding(self.csv)

      self.assertEqual(received.to_json(), expected.to_json())

    def test_label_encoding(self):
      received = adult.label_encoding(self.csv)

      le = LabelEncoder()
      encoded_labels = le.fit_transform(self.csv["class"])
      expected = pd.Series(encoded_labels)

      self.assertEqual(received.to_json(), expected.to_json())

    def test_dt_predict(self):
      X_train = adult.one_hot_encoding(self.csv)
      y_train = adult.label_encoding(self.csv)

      received = adult.dt_predict(X_train, y_train)
      expected = pd.Series(DecisionTreeClassifier().fit(X_train, y_train).predict(X_train))

      self.assertEqual(received.to_json(), expected.to_json())

    def test_dt_error_rate(self):
      new_df = adult.data_frame_without_missing_values(self.csv)
      X_train = adult.one_hot_encoding(new_df)
      y_train = adult.label_encoding(new_df)
      y_pred = pd.Series(DecisionTreeClassifier().fit(X_train, y_train).predict(X_train))

      received = adult.dt_error_rate(y_pred, y_train)
      expected = 0.08279156162929546


      self.assertEqual(received, expected)


# driver code
if __name__ == '__main__':
    unittest.main()