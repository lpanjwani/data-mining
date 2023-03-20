# import library
import unittest
import pandas as pd

# Import Test File
import coronavirus_tweets

# create a class
class TestCoronavirusTweets(unittest.TestCase):
    def setUp(self):
      self.csv = pd.read_csv('data/coronavirus_tweets.csv', encoding='latin-1')

    def test_read_csv_3(self):
      expected = self.csv
      received = coronavirus_tweets.read_csv_3('data/coronavirus_tweets.csv')

      self.assertEqual(expected.to_json(), received.to_json())

    def test_get_sentiments(self):
      expected = ['Neutral', 'Positive', 'Extremely Negative', 'Negative', 'Extremely Positive']
      received = coronavirus_tweets.get_sentiments(self.csv)

      self.assertEqual(expected, list(received))

    def test_second_most_popular_sentiment(self):
      expected = 'Negative'
      received = coronavirus_tweets.second_most_popular_sentiment(self.csv)

      self.assertEqual(expected, received)

    def test_date_most_popular_tweets(self):
      expected = '25-03-2020'
      received = coronavirus_tweets.date_most_popular_tweets(self.csv)

      self.assertEqual(expected, received)

    def test_lower_case(self):
      expected = self.csv['OriginalTweet'].str.lower()
      received = coronavirus_tweets.lower_case(self.csv)

      self.assertEqual(expected.to_json(), received.to_json())

    def test_remove_non_alphabetic_chars(self):
      expected = self.csv['OriginalTweet'].str.replace('[^a-zA-Z\s]', ' ')
      received = coronavirus_tweets.remove_non_alphabetic_chars(self.csv)

      self.assertEqual(expected.to_json(), received.to_json())

    def test_remove_multiple_consecutive_whitespaces(self):
      expected = self.csv['OriginalTweet'].str.replace('\s+', ' ')
      received = coronavirus_tweets.remove_multiple_consecutive_whitespaces(self.csv)

      self.assertEqual(expected.to_json(), received.to_json())

    def test_tokenize(self):
      expected = self.csv['OriginalTweet'].str.split()
      received = coronavirus_tweets.tokenize(self.csv)

      self.assertEqual(expected.to_json(), received.to_json())

    def test_count_words_with_repetitions(self):
      expected = 1255301 # TODO: Verify
      received = coronavirus_tweets.count_words_with_repetitions(self.csv)

      self.assertEqual(expected, received)

    def test_count_words_without_repetitions(self):
      expected = 63 # TODO: Verify
      received = coronavirus_tweets.count_words_without_repetitions(self.csv)

      self.assertEqual(expected, received)

    def test_frequent_words(self):
      expected = ['the', 'to', 'and', 'of', 'on']
      received = coronavirus_tweets.frequent_words(self.csv, 5)

      self.assertEqual(expected, received)

    def test_remove_stopwords(self):
      expected = self.csv['OriginalTweet'].str.replace('\bthe\b', '')
      received = coronavirus_tweets.remove_stop_words(self.csv)

      self.assertEqual(expected.to_json(), received.to_json())

    def test_stemming(self):
      expected = self.csv['OriginalTweet'].str.replace('ing', '')
      received = coronavirus_tweets.stemming(self.csv)

      self.assertEqual(expected.to_json(), received.to_json())




# driver code
if __name__ == '__main__':
    unittest.main()