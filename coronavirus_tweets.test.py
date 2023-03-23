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
      expected = self.csv.copy()
      expected['OriginalTweet'] = expected['OriginalTweet'].str.lower()

      received = coronavirus_tweets.lower_case(self.csv)

      self.assertEqual(expected.to_json(), received.to_json())

    def test_remove_non_alphabetic_chars(self):
      expected = self.csv.copy()
      expected['OriginalTweet'] = expected['OriginalTweet'].str.replace('[^a-zA-Z\s]', ' ', regex=True)

      received = coronavirus_tweets.remove_non_alphabetic_chars(self.csv)

      self.assertEqual(expected.to_json(), received.to_json())

    def test_remove_multiple_consecutive_whitespaces(self):
      expected = self.csv.copy()
      expected['OriginalTweet'] = expected['OriginalTweet'].str.replace('\s+', ' ', regex=True)

      received = coronavirus_tweets.remove_multiple_consecutive_whitespaces(self.csv)

      self.assertEqual(expected.to_json(), received.to_json())

    def test_tokenize(self):
      expected = self.csv.copy()
      expected['OriginalTweet'] = expected['OriginalTweet'].str.split()

      received = coronavirus_tweets.tokenize(self.csv)

      self.assertEqual(expected.to_json(), received.to_json())

    def test_count_words_with_repetitions(self):
      expected = 1350959

      input_args = coronavirus_tweets.lower_case(self.csv)
      input_args = coronavirus_tweets.remove_non_alphabetic_chars(input_args)
      input_args = coronavirus_tweets.remove_multiple_consecutive_whitespaces(input_args)
      input_args = coronavirus_tweets.tokenize(input_args)

      received = coronavirus_tweets.count_words_with_repetitions(input_args)

      self.assertEqual(expected, received)

    def test_count_words_without_repetitions(self):
      expected = 80071

      input_args = coronavirus_tweets.lower_case(self.csv)
      input_args = coronavirus_tweets.remove_non_alphabetic_chars(input_args)
      input_args = coronavirus_tweets.remove_multiple_consecutive_whitespaces(input_args)
      input_args = coronavirus_tweets.tokenize(input_args)

      received = coronavirus_tweets.count_words_without_repetitions(input_args)

      self.assertEqual(expected, received)

    def test_frequent_words(self):
      expected = ['the', 'to', 't', 'co', 'and', 'https', 'covid', 'of', 'a', 'in']

      input_args = coronavirus_tweets.lower_case(self.csv)
      input_args = coronavirus_tweets.remove_non_alphabetic_chars(input_args)
      input_args = coronavirus_tweets.remove_multiple_consecutive_whitespaces(input_args)
      input_args = coronavirus_tweets.tokenize(input_args)

      received = coronavirus_tweets.frequent_words(input_args, 10)

      self.assertEqual(expected, received)

    def test_remove_stopwords(self):
      expected_length = 11
      expected_values = ['menyrbie', 'phil', 'gahan', 'chrisitv', 'https', 'ifz', 'fan', 'https', 'ghgfzcc', 'https', 'nlzdxno']

      input_args = coronavirus_tweets.lower_case(self.csv)
      input_args = coronavirus_tweets.remove_non_alphabetic_chars(input_args)
      input_args = coronavirus_tweets.remove_multiple_consecutive_whitespaces(input_args)
      input_args = coronavirus_tweets.tokenize(input_args)

      received = coronavirus_tweets.remove_stop_words(input_args)
      received_first_row = received['OriginalTweet'].iloc[0]

      self.assertEqual(len(received_first_row), expected_length)
      self.assertEqual(received_first_row, expected_values)

    def test_stemming(self):
      expected_value = ['menyrbi', 'phil', 'gahan', 'chrisitv', 'http', 'ifz', 'fan', 'http', 'ghgfzcc', 'http', 'nlzdxno']
      expected_length = 11

      input_args = coronavirus_tweets.lower_case(self.csv)
      input_args = coronavirus_tweets.remove_non_alphabetic_chars(input_args)
      input_args = coronavirus_tweets.remove_multiple_consecutive_whitespaces(input_args)
      input_args = coronavirus_tweets.tokenize(input_args)
      input_args = coronavirus_tweets.remove_stop_words(input_args)

      received = coronavirus_tweets.stemming(input_args)
      received_value = received['OriginalTweet'].iloc[0]

      self.assertEqual(expected_value, received_value)
      self.assertEqual(len(received_value), expected_length)

    def test_mnb_predict(self):
      expected = len(self.csv["Sentiment"])
      received = coronavirus_tweets.mnb_predict(self.csv)

      self.assertEqual(expected, len(received))

    def test_mnb_accuracy(self):
      expected = 0.995

      predict_values = coronavirus_tweets.mnb_predict(self.csv)

      received = coronavirus_tweets.mnb_accuracy(predict_values, self.csv["Sentiment"].values)

      self.assertEqual(expected, received)



# driver code
if __name__ == '__main__':
    unittest.main()