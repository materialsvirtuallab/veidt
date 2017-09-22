from veidt.elsie import similarity_measures
import sklearn.metrics.pairwise as smp
import numpy as np
import unittest


class SimilarityMeasureTest(unittest.TestCase):
    def setUp(self):
        self.v1, self.v2 = np.array([1, 0]), np.array([-1, 0])
        self.v3, self.v4 = np.array([0, 1]), np.array([1, 0])
        self.v5, self.v6 = np.array([0.5, 0.5]), np.array([0.8, 0.2])
        self.v7, self.v8 = np.array([1, 0, 0]), np.array([0.3, 0.7])
        self.v9, self.v10 = np.array([0.2, 0.3]), np.array([0.4, 0.3])

    def test_normal_fail(self):
        cos_simi = getattr(similarity_measures, 'Cosine')
        self.assertRaisesRegex(ValueError, "Spectrum 1 .* not.*normalized", cos_simi, coeff_1=self.v9,
                               coeff_2=self.v1)
        self.assertRaisesRegex(ValueError, "Spectrum 1 .* not.*normalized", cos_simi, coeff_1=self.v9,
                               coeff_2=self.v10)
        self.assertRaisesRegex(ValueError, "Spectrum 2 .* not.*normalized", cos_simi, coeff_1=self.v1,
                               coeff_2=self.v10)
        euc_simi = getattr(similarity_measures, 'Euclidean')
        self.assertRaisesRegex(ValueError, "Spectrum 1 .* not.*normalized", euc_simi, coeff_1=self.v9,
                               coeff_2=self.v1)
        self.assertRaisesRegex(ValueError, "Spectrum 1 .* not.*normalized", euc_simi, coeff_1=self.v9,
                               coeff_2=self.v10)
        self.assertRaisesRegex(ValueError, "Spectrum 2 .* not.*normalized", euc_simi, coeff_1=self.v1,
                               coeff_2=self.v10)
        ruzi_simi = getattr(similarity_measures, 'Ruzicka')
        self.assertRaisesRegex(ValueError, "Spectrum 1 .* not.*normalized", ruzi_simi, coeff_1=self.v9,
                               coeff_2=self.v1)
        self.assertRaisesRegex(ValueError, "Spectrum 1 .* not.*normalized", ruzi_simi, coeff_1=self.v9,
                               coeff_2=self.v10)
        self.assertRaisesRegex(ValueError, "Spectrum 2 .* not.*normalized", ruzi_simi, coeff_1=self.v1,
                               coeff_2=self.v10)

    def test_cosine_similarity(self):
        cos_simi = getattr(similarity_measures, 'Cosine')
        self.assertRaisesRegex(ValueError, "different wavelength", cos_simi, coeff_1=self.v1, coeff_2=self.v7)
        self.assertTrue(cos_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertEqual(cos_simi(self.v1, self.v5).similarity_measure(),
                         smp.cosine_similarity(self.v1.reshape(1, -1), self.v5.reshape(1, -1)))
        self.assertEqual(cos_simi(self.v1, self.v3).similarity_measure(),
                         smp.cosine_similarity(self.v1.reshape(1, -1), self.v3.reshape(1, -1)))
        self.assertEqual(cos_simi(self.v1, self.v6).similarity_measure(),
                         smp.cosine_similarity(self.v1.reshape(1, -1), self.v6.reshape(1, -1)))
        self.assertEqual(cos_simi(self.v1, self.v6).__str__(), "CosineSimilarity")

    def test_euclidean_similarity(self):
        euc_simi = getattr(similarity_measures, 'Euclidean')
        self.assertEqual(euc_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertEqual(euc_simi(self.v1, self.v4).similarity_measure('exp'), 1.0)
        self.assertEqual(euc_simi(self.v1, self.v4).__str__(), 'EuclideanSimilarity')
        self.assertEqual(euc_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertEqual(euc_simi(self.v1, self.v3).distance_measure(),
                         smp.euclidean_distances(self.v1.reshape(1, -1), self.v3.reshape(1, -1)))
        self.assertEqual(euc_simi(self.v1, self.v5).distance_measure(),
                         smp.euclidean_distances(self.v1.reshape(1, -1), self.v5.reshape(1, -1)))
        self.assertTrue(np.allclose(euc_simi(self.v1, self.v6).distance_measure(),
                         smp.euclidean_distances(self.v1.reshape(1, -1), self.v6.reshape(1, -1))))

    def test_pearson_correlation(self):
        pear_simi = getattr(similarity_measures, 'PearsonCorrMeasure')
        self.assertEqual(pear_simi(self.v1, self.v2).similarity_measure(), -1.0)
        self.assertEqual(pear_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertEqual(pear_simi(self.v1, self.v4).__str__(), "PearsonSimilarity")

    def test_ruzicka_similarity(self):
        ruzi_simi = getattr(similarity_measures, 'Ruzicka')
        self.assertEqual(ruzi_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertEqual(ruzi_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertEqual(ruzi_simi(self.v1, self.v4).__str__(), "RuzickaSimilarity")

    def test_cityblock_similarity(self):
        cityblock_simi = getattr(similarity_measures, 'Cityblock')
        self.assertEqual(cityblock_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertEqual(cityblock_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertTrue(cityblock_simi(self.v1, self.v5).similarity_measure() < 1.0)
        self.assertTrue(cityblock_simi(self.v1, self.v5).similarity_measure() > 0.0)
        self.assertEqual(cityblock_simi(self.v1, self.v3).__str__(), "CityblockSimilarity")

    def test_minkowski_similarity(self):
        minkowski_simi = getattr(similarity_measures, 'Minkowski')
        self.assertEqual(minkowski_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertEqual(minkowski_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertTrue(minkowski_simi(self.v1, self.v5).similarity_measure() < 1.0)
        self.assertTrue(minkowski_simi(self.v1, self.v5).similarity_measure() > 0.0)
        self.assertEqual(minkowski_simi(self.v1, self.v3).__str__(), "MinkowskiSimilarity")

    def test_chebyshev_similarity(self):
        chebyshev_simi = getattr(similarity_measures, 'Chebyshev')
        self.assertEqual(chebyshev_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertEqual(chebyshev_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertTrue(chebyshev_simi(self.v1, self.v5).similarity_measure() < 1.0)
        self.assertTrue(chebyshev_simi(self.v1, self.v5).similarity_measure() > 0.0)
        self.assertEqual(chebyshev_simi(self.v1, self.v3).__str__(), "ChebyshevSimilarity")

    def test_sorensen_similarity(self):
        sorensen_simi = getattr(similarity_measures, 'Sorensen')
        self.assertEqual(sorensen_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertEqual(sorensen_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertTrue(sorensen_simi(self.v1, self.v5).similarity_measure() < 1.0)
        self.assertTrue(sorensen_simi(self.v1, self.v5).similarity_measure() > 0.0)
        self.assertTrue(sorensen_simi(self.v1, self.v8).similarity_measure() < 1.0)
        self.assertTrue(sorensen_simi(self.v1, self.v8).similarity_measure() > 0.0)
        self.assertEqual(sorensen_simi(self.v1, self.v3).__str__(), "SorensenSimilarity")

    def test_kulczynski_similarity(self):
        kulczynski_simi = getattr(similarity_measures, 'Kulczynski')
        self.assertTrue(kulczynski_simi(self.v1, self.v5).similarity_measure() < 1.0)
        self.assertTrue(kulczynski_simi(self.v1, self.v5).similarity_measure() > 0.0)
        self.assertTrue(kulczynski_simi(self.v1, self.v8).similarity_measure() < 1.0)
        self.assertTrue(kulczynski_simi(self.v1, self.v8).similarity_measure() > 0.0)
        self.assertEqual(kulczynski_simi(self.v1, self.v3).__str__(), "KulczynskiSimilarity")

    def test_lorentzian_similarity(self):
        lorentzian_simi = getattr(similarity_measures, 'Lorentzian')
        self.assertEqual(lorentzian_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertEqual(lorentzian_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertTrue(lorentzian_simi(self.v1, self.v5).similarity_measure() < 1.0)
        self.assertTrue(lorentzian_simi(self.v1, self.v5).similarity_measure() > 0.0)
        self.assertTrue(lorentzian_simi(self.v1, self.v8).similarity_measure() < 1.0)
        self.assertTrue(lorentzian_simi(self.v1, self.v8).similarity_measure() > 0.0)
        self.assertEqual(lorentzian_simi(self.v1, self.v3).__str__(), "LorentzianSimilarity")

    def test_intersection_similarity(self):
        intersection_simi = getattr(similarity_measures, 'Intersection')
        self.assertEqual(intersection_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertEqual(intersection_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertTrue(intersection_simi(self.v1, self.v5).similarity_measure() < 1.0)
        self.assertTrue(intersection_simi(self.v1, self.v5).similarity_measure() > 0.0)
        self.assertTrue(intersection_simi(self.v1, self.v8).similarity_measure() < 1.0)
        self.assertTrue(intersection_simi(self.v1, self.v8).similarity_measure() > 0.0)
        self.assertEqual(intersection_simi(self.v1, self.v3).__str__(), "IntersectionSimilarity")

    def test_czekanowski_similarity(self):
        czekanowski_simi = getattr(similarity_measures, 'Czekanowski')
        self.assertEqual(czekanowski_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertEqual(czekanowski_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertTrue(czekanowski_simi(self.v1, self.v5).similarity_measure() < 1.0)
        self.assertTrue(czekanowski_simi(self.v1, self.v5).similarity_measure() > 0.0)
        self.assertTrue(czekanowski_simi(self.v1, self.v8).similarity_measure() < 1.0)
        self.assertTrue(czekanowski_simi(self.v1, self.v8).similarity_measure() > 0.0)
        self.assertEqual(czekanowski_simi(self.v1, self.v3).__str__(), "CzekanowskiSimilarity")

    def test_motyka_similarity(self):
        motyka_simi = getattr(similarity_measures, 'Motyka')
        self.assertEqual(motyka_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertEqual(motyka_simi(self.v1, self.v4).similarity_measure(), 0.5)
        self.assertTrue(motyka_simi(self.v1, self.v5).similarity_measure() < 0.5)
        self.assertTrue(motyka_simi(self.v1, self.v5).similarity_measure() > 0.0)
        self.assertTrue(motyka_simi(self.v1, self.v8).similarity_measure() < 0.5)
        self.assertTrue(motyka_simi(self.v1, self.v8).similarity_measure() > 0.0)
        self.assertEqual(motyka_simi(self.v1, self.v3).__str__(), "MotykaSimilarity")

    def test_tanimoto_similarity(self):
        tanimoto_simi = getattr(similarity_measures, 'Tanimoto')
        self.assertEqual(tanimoto_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertEqual(tanimoto_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertTrue(tanimoto_simi(self.v1, self.v5).similarity_measure() < 1.0)
        self.assertTrue(tanimoto_simi(self.v1, self.v5).similarity_measure() > 0.0)
        self.assertTrue(tanimoto_simi(self.v1, self.v8).similarity_measure() < 1.0)
        self.assertTrue(tanimoto_simi(self.v1, self.v8).similarity_measure() > 0.0)
        self.assertEqual(tanimoto_simi(self.v1, self.v3).__str__(), "TanimotoSimilarity")

    def test_innerproduct_similarity(self):
        innerproduct_simi = getattr(similarity_measures, 'InnerProduct')
        self.assertEqual(innerproduct_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertEqual(innerproduct_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertTrue(innerproduct_simi(self.v1, self.v5).similarity_measure() < 1.0)
        self.assertTrue(innerproduct_simi(self.v1, self.v5).similarity_measure() > 0.0)
        self.assertTrue(innerproduct_simi(self.v1, self.v8).similarity_measure() < 1.0)
        self.assertTrue(innerproduct_simi(self.v1, self.v8).similarity_measure() > 0.0)
        self.assertEqual(innerproduct_simi(self.v1, self.v3).__str__(), "InnerProductSimilarity")

    def test_harmonicmean_similarity(self):
        harmonicmean_simi = getattr(similarity_measures, 'HarmonicMean')
        self.assertEqual(harmonicmean_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertEqual(harmonicmean_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertTrue(harmonicmean_simi(self.v1, self.v5).similarity_measure() < 1.0)
        self.assertTrue(harmonicmean_simi(self.v1, self.v5).similarity_measure() > 0.0)
        self.assertTrue(harmonicmean_simi(self.v1, self.v8).similarity_measure() < 1.0)
        self.assertTrue(harmonicmean_simi(self.v1, self.v8).similarity_measure() > 0.0)
        self.assertEqual(harmonicmean_simi(self.v1, self.v3).__str__(), "HarmonicMeanSimilarity")

    def test_jaccard_similarity(self):
        jaccard_simi = getattr(similarity_measures, 'Jaccard')
        self.assertEqual(jaccard_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertEqual(jaccard_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertTrue(jaccard_simi(self.v1, self.v5).similarity_measure() < 1.0)
        self.assertTrue(jaccard_simi(self.v1, self.v5).similarity_measure() > 0.0)
        self.assertTrue(jaccard_simi(self.v1, self.v8).similarity_measure() < 1.0)
        self.assertTrue(jaccard_simi(self.v1, self.v8).similarity_measure() > 0.0)
        self.assertEqual(jaccard_simi(self.v1, self.v3).__str__(), "JaccardSimilarity")

    def test_dice_similarity(self):
        dice_simi = getattr(similarity_measures, 'Dice')
        self.assertEqual(dice_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertEqual(dice_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertTrue(dice_simi(self.v1, self.v5).similarity_measure() < 1.0)
        self.assertTrue(dice_simi(self.v1, self.v5).similarity_measure() > 0.0)
        self.assertTrue(dice_simi(self.v1, self.v8).similarity_measure() < 1.0)
        self.assertTrue(dice_simi(self.v1, self.v8).similarity_measure() > 0.0)
        self.assertEqual(dice_simi(self.v1, self.v3).__str__(), "DiceSimilarity")

    def test_fidelity_similarity(self):
        fidelity_simi = getattr(similarity_measures, 'Fidelity')
        self.assertEqual(fidelity_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertEqual(fidelity_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertTrue(fidelity_simi(self.v1, self.v5).similarity_measure() < 1.0)
        self.assertTrue(fidelity_simi(self.v1, self.v5).similarity_measure() > 0.0)
        self.assertTrue(fidelity_simi(self.v1, self.v8).similarity_measure() < 1.0)
        self.assertTrue(fidelity_simi(self.v1, self.v8).similarity_measure() > 0.0)
        self.assertEqual(fidelity_simi(self.v1, self.v3).__str__(), "FidelitySimilarity")

    def test_hellinger_similarity(self):
        hellinger_simi = getattr(similarity_measures, 'Hellinger')
        self.assertEqual(hellinger_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertEqual(hellinger_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertTrue(hellinger_simi(self.v1, self.v5).similarity_measure() < 1.0)
        self.assertTrue(hellinger_simi(self.v1, self.v5).similarity_measure() > 0.0)
        self.assertTrue(hellinger_simi(self.v1, self.v8).similarity_measure() < 1.0)
        self.assertTrue(hellinger_simi(self.v1, self.v8).similarity_measure() > 0.0)
        self.assertEqual(hellinger_simi(self.v1, self.v3).__str__(), "HellingerSimilarity")

    def test_matusita_similarity(self):
        matusita_simi = getattr(similarity_measures, 'Matusita')
        self.assertEqual(matusita_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertEqual(matusita_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertTrue(matusita_simi(self.v1, self.v5).similarity_measure() < 1.0)
        self.assertTrue(matusita_simi(self.v1, self.v5).similarity_measure() > 0.0)
        self.assertTrue(matusita_simi(self.v1, self.v8).similarity_measure() < 1.0)
        self.assertTrue(matusita_simi(self.v1, self.v8).similarity_measure() > 0.0)
        self.assertEqual(matusita_simi(self.v1, self.v3).__str__(), "MatusitaSimilarity")

    def test_squaredchord_similarity(self):
        squachord_simi = getattr(similarity_measures, 'Squaredchord')
        self.assertEqual(squachord_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertEqual(squachord_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertTrue(squachord_simi(self.v1, self.v5).similarity_measure() < 1.0)
        self.assertTrue(squachord_simi(self.v1, self.v5).similarity_measure() > 0.0)
        self.assertTrue(squachord_simi(self.v1, self.v8).similarity_measure() < 1.0)
        self.assertTrue(squachord_simi(self.v1, self.v8).similarity_measure() > 0.0)
        self.assertEqual(squachord_simi(self.v1, self.v3).__str__(), "SquaredchordSimilarity")

    def test_squaredeuclidean_similarity(self):
        squaeucli_simi = getattr(similarity_measures, 'SquaredEuclidean')
        self.assertAlmostEqual(squaeucli_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertAlmostEqual(squaeucli_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertTrue(squaeucli_simi(self.v1, self.v5).similarity_measure() < 1.0)
        self.assertTrue(squaeucli_simi(self.v1, self.v5).similarity_measure() > 0.0)
        self.assertTrue(squaeucli_simi(self.v1, self.v8).similarity_measure() < 1.0)
        self.assertTrue(squaeucli_simi(self.v1, self.v8).similarity_measure() > 0.0)
        self.assertEqual(squaeucli_simi(self.v1, self.v3).__str__(), "SquaredEuclideanSimilarity")

    def test_squaredchisquare_similarity(self):
        squachisquare_simi = getattr(similarity_measures, 'SquaredChiSquare')
        self.assertAlmostEqual(squachisquare_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertAlmostEqual(squachisquare_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertTrue(squachisquare_simi(self.v1, self.v5).similarity_measure() < 1.0)
        self.assertTrue(squachisquare_simi(self.v1, self.v5).similarity_measure() > 0.0)
        self.assertTrue(squachisquare_simi(self.v1, self.v8).similarity_measure() < 1.0)
        self.assertTrue(squachisquare_simi(self.v1, self.v8).similarity_measure() > 0.0)
        self.assertEqual(squachisquare_simi(self.v1, self.v3).__str__(), "SquaredChiSquare Similarity")

    def test_probsymmchi_similarity(self):
        probsymmchi_simi = getattr(similarity_measures, 'ProbabilisticSymmetricChiS')
        self.assertAlmostEqual(probsymmchi_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertAlmostEqual(probsymmchi_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertTrue(probsymmchi_simi(self.v1, self.v5).similarity_measure() < 1.0)
        self.assertTrue(probsymmchi_simi(self.v1, self.v5).similarity_measure() > 0.0)
        self.assertTrue(probsymmchi_simi(self.v1, self.v8).similarity_measure() < 1.0)
        self.assertTrue(probsymmchi_simi(self.v1, self.v8).similarity_measure() > 0.0)
        self.assertEqual(probsymmchi_simi(self.v1, self.v3).__str__(),
                         "Probabilistic Symmetric ChiSquare Similarity")

    def test_avgl1linf_similarity(self):
        avgl1linf_simi = getattr(similarity_measures, 'AvgL1Linf')
        self.assertAlmostEqual(avgl1linf_simi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertAlmostEqual(avgl1linf_simi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertTrue(avgl1linf_simi(self.v1, self.v5).similarity_measure() < 1.0)
        self.assertTrue(avgl1linf_simi(self.v1, self.v5).similarity_measure() > 0.0)
        self.assertTrue(avgl1linf_simi(self.v1, self.v8).similarity_measure() < 1.0)
        self.assertTrue(avgl1linf_simi(self.v1, self.v8).similarity_measure() > 0.0)
        self.assertEqual(avgl1linf_simi(self.v1, self.v3).__str__(),
                         "Average L1 L_inf Similarity")

    def test_minsymmetricchi_similarity(self):
        minsymmetricchi = getattr(similarity_measures, 'MinSymmetricChi')
        self.assertAlmostEqual(minsymmetricchi(self.v1, self.v3).similarity_measure(), 0.0)
        self.assertAlmostEqual(minsymmetricchi(self.v1, self.v4).similarity_measure(), 1.0)
        self.assertTrue(minsymmetricchi(self.v1, self.v5).similarity_measure() < 1.0)
        self.assertTrue(minsymmetricchi(self.v1, self.v5).similarity_measure() > 0.0)
        self.assertTrue(minsymmetricchi(self.v1, self.v8).similarity_measure() < 1.0)
        self.assertTrue(minsymmetricchi(self.v1, self.v8).similarity_measure() > 0.0)
        self.assertEqual(minsymmetricchi(self.v1, self.v3).__str__(),
                         "minsymmetric Chisquare Similarity")
