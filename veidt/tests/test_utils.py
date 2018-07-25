import unittest
import numpy as np
from veidt.utils.general_utils import serialize_veidt_object, deserialize_veidt_object
import os

file_path = os.path.dirname(__file__)


def test_func():
    return 1


class DummyClass:
    def __init__(self):
        self.name = 'dummy'

    def get_config(self):
        return {"config": "Dummyclass config"}

class TestGeneralUtil(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x1 = np.array([1, 2, 3])
        cls.x2 = np.array([4, 5, 6])
        cls.x3 = np.array([1, 1, 1])
        cls.x4 = np.array([1, 2, 3])

    def test_serialization(self):
        self.assertEqual(serialize_veidt_object(test_func), "test_func")
        self.assertEqual(serialize_veidt_object(DummyClass())['class_name'], "DummyClass")
        self.assertIsNone(serialize_veidt_object(None))


    def test_deserialization(self):
        self.assertEqual(1, deserialize_veidt_object('test_func', module_objects=globals())())
        self.assertIsInstance(deserialize_veidt_object({"class_name": "DummyClass",
                                                            "config": {}}, module_objects=globals()), DummyClass)



if __name__ == '__main__':
    unittest.main()

