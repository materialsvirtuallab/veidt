import unittest
import numpy as np
from veidt.abstract import Describer, Model
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

file_path = os.path.dirname(__file__)


class DummyDescriber(Describer):
    def describe(self, obj):
        return pd.Series(np.sum(obj))


class DummyModel(Model):
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, x, y):
        self.model.fit(x, y)
        return self

    def predict(self, x):
        return self.model.predict(x)



class TestDescrber(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dd = DummyDescriber()

    def test_fit(self):
        dd2 = self.dd.fit([1, 2, 3])
        self.assertEqual(dd2, self.dd)

    def test_describe(self):
        result = self.dd.describe([1, 2, 3])
        self.assertEqual(result.values[0], 6)

    def test_describe_all(self):
        results = self.dd.describe_all([[1, 1, 1], [2, 2, 2]])
        self.assertListEqual(list(results.shape), [2])
        results_transform = self.dd.transform([[1, 1, 1], [2, 2, 2]])
        self.assertEqual(9, np.sum(results_transform))


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DummyModel()

    def test_fit(self):
        model_dummy = self.model.fit([[1, 2], [3, 4]], [[3], [7]])
        self.assertEqual(model_dummy, self.model)

    def test_predict(self):
        self.model.fit([[1, 2], [3, 4]], [[3], [7]])
        result = self.model.predict([[1, 5]])
        self.assertEqual(result[0], 6)

    def test_evaluate(self):
        self.model.fit([[1, 2], [3, 4]], [[3], [7]])
        error = self.model.evaluate([[1, 2], [3, 4]], [[4, 8]])
        # print(error)
        self.assertAlmostEqual(error['mae'][0], 1)

if __name__ == '__main__':
    unittest.main()

