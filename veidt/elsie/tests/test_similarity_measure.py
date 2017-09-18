from veidt.elsie import similarity_measures
import sklearn.metrics.pairwise as smp
import numpy as np
import unittest


class SimilarityMeasureTest(unittest.TestCase):
    def setUp(self):
        self.v1, self.v2 = np.array([1, 0]), np.array([-1, 0])
        self.v3, self.v4 = np.array([0, 1]), np.array([1, 0])
        self.v5, self.v6 = np.array([0.5, 0.5]), np.array([-0.5, -0.5])

    def test_cosine_similarity(self):
        cos_simi = getattr(similarity_measures, 'Cosine')
        self.assertTrue(cos_simi(self.v1, self.v2).similarity_measure(), -1.0)
        self.assertTrue(cos_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertEqual(cos_simi(self.v1, self.v5).similarity_measure(),
                         smp.cosine_similarity(self.v1.reshape(1, -1), self.v5.reshape(1, -1)))
        self.assertEqual(cos_simi(self.v1, self.v3).similarity_measure(),
                         smp.cosine_similarity(self.v1.reshape(1, -1), self.v3.reshape(1, -1)))
        self.assertEqual(cos_simi(self.v1, self.v6).similarity_measure(),
                         smp.cosine_similarity(self.v1.reshape(1, -1), self.v6.reshape(1, -1)))

    def test_euclidean_similarity(self):
        euc_simi = getattr(similarity_measures, 'Euclidean')
        self.assertEqual(euc_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertEqual(euc_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertEqual(euc_simi(self.v1, self.v3).distance_measure(),
                         smp.euclidean_distances(self.v1.reshape(1, -1), self.v3.reshape(1, -1)))
        self.assertEqual(euc_simi(self.v1, self.v5).distance_measure(),
                         smp.euclidean_distances(self.v1.reshape(1, -1), self.v5.reshape(1, -1)))
        self.assertEqual(euc_simi(self.v1, self.v6).distance_measure(),
                         smp.euclidean_distances(self.v1.reshape(1, -1), self.v6.reshape(1, -1)))

    def test_pearson_correlation(self):
        pear_simi = getattr(similarity_measures, 'PearsonCorrMeasure')
        self.assertEqual(pear_simi(self.v1, self.v2).similarity_measure(), -1.0)
        self.assertEqual(pear_simi(self.v1, self.v4).similarity_measure(), 1.0)

    def test_ruzicka_similarity(self):
        ruzi_simi = getattr(similarity_measures, 'Ruzicka')
        self.assertEqual(ruzi_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertEqual(ruzi_simi(self.v1, self.v4).similarity_measure(), 1.0)
