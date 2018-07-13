import unittest
import os
from bson.json_util import loads
import numpy as np
from pymatgen.core import Structure
from veidt.potential.agni import AGNIPotentialVeidt

pjoin = os.path.join
dirname = os.path.dirname(__file__)

class TestAgniVeidt(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.file = pjoin(dirname, "data", "Ni.json")
        with open(cls.file, 'r') as f:
            cls.data = loads(f.read())
        cls.structures = [Structure.from_dict(i['structure']) for i in cls.data[:5]]
        cls.forces = np.concatenate([np.array(i['outputs']['forces']).reshape((-1, 1)) for i in cls.data[:5]], axis=0)
        cls.test_structures = [Structure.from_dict(i['structure']) for i in cls.data[5:10]]
        cls.test_forces = np.concatenate([np.array(i['outputs']['forces']).reshape((-1, 1)) for i in cls.data[5:10]], axis=0)
        cls.model = AGNIPotentialVeidt(element="Ni", sigma=0.0001)

    def test_feature_calculation(self):
        features = self.model.describer.transform(self.structures)
        self.assertEqual(features.shape[1], 8)

    def test_fit(self):
        self.model.train(self.structures, self.forces)
        self.assertEqual(self.model.xu.shape[1], 8)

    def test_lammps_calc(self):
        out = self.model.predict_structures(self.structures[:10])
        self.assertListEqual(list(out.shape), [3240, 1])
