__author__ = "Chen Zheng, Hanmei Tang"
__copyright__ = "Copyright 2012, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Chen Zheng"
__email__ = "chz022@eng.ucsd.edu"
__date__ = "May 11, 2017"

import sklearn.metrics.pairwise as smp
from monty.json import MSONable
from scipy.stats import pearsonr
import scipy.spatial.distance as spd
import numpy as np


class SimilarityMeasure(MSONable):
    """
    Abstract class used to define the various methods that must be implemented by
    similarity measurement class. All measurement class must also implement the
    standard MSONable as_dict() and from_dict() API
    """

    def __init__(self, coeff_1, coeff_2):
        """
        :param coeff_1: numpy array with dimension (n, 1), n corresponding to number of
                        wavelength, column corresponding to the absorption coefficiency
                        The spectrum need to be normalized to obtain meaningful result, i.e.
                        the under curve area of spectrum need equal to 1
        :param coeff_2: numpy array with dimension (n, 1). The row and column definition
                        is the same as spectrum 1. The spectrum need to be normalized to
                        obtain meaningful result, i.e. the under curve area of spectrum need equal to 1
        """
        if len(coeff_1) != len(coeff_2):
            raise ValueError('Two spectrum have different wavelength number')

        self.coeff_1 = coeff_1
        self.coeff_2 = coeff_2
        self.d_max = None

    def normalize_spectrum(self, spec_1):

        raise NotImplementedError()

    def distance_measure(self):
        """
        Compute the distance measures of two spectrum
        :return: distance measure between two spectrum
        """
        raise NotImplementedError()

    def similarity_measure(self, dist_conversion='bin'):
        """
        Compute the similarity measure of two spectrum

        :param: dist_conversion: algorithm used to convert distance measure to similarity
                exponential conversion are more sensitive for detecting extremely fine changes
                in spectrum difference
        :return: similarity measure between two spectrum
        """
        coeff_dist = self.distance_measure()

        if dist_conversion == 'bin':
            simi_measure = (1 - coeff_dist / self.d_max)
        elif dist_conversion == 'exp':
            simi_measure = np.exp(-(coeff_dist / (self.d_max - coeff_dist)))
        return simi_measure

    def as_dict(self):
        return {"init_args": {"absorbing_coeff1": self.coeff_1,
                              "absorbing_coeff2": self.coeff_2},
                "version": __version__,
                "@module": self.__class__.__module__,
                "@class": self.__class__.__name__}

    @classmethod
    def from_dict(cls, d):
        return cls(**d["init_args"])


class Euclidean(SimilarityMeasure):
    def __init__(self, coeff_1, coeff_2):
        """
        Class to calculate the Euclidean similarity
        """
        super().__init__(coeff_1, coeff_2)
        self.d_max = np.sqrt(2)

    def distance_measure(self):
        return spd.euclidean(self.coeff_1, self.coeff_2)

    def __str__(self):
        return "EuclideanSimilarity"


class Cityblock(SimilarityMeasure):
    """
    Cityblock similarity to calculate the Cityblock, i.e. Manhattan, similarity
    """

    def __init__(self, coeff_1, coeff_2):
        super().__init__(coeff_1, coeff_2)
        self.d_max = 2

    def distance_measure(self):
        return spd.cityblock(self.coeff_1, self.coeff_2)

    def __str__(self):
        return "CityblockSimilarity"


class Minkowski(SimilarityMeasure):
    """
    Minkowski similarity to calculate the Cityblock, i.e. Manhattan, similarity
    :param p: The order of the norm of the difference
    """

    def __init__(self, coeff_1, coeff_2, p=4):
        super().__init__(coeff_1, coeff_2)
        self.p = p
        self.d_max = np.power(2, 1.0 / p)

    def distance_measure(self):
        return spd.minkowski(self.coeff_1, self.coeff_2, self.p)

    def __str__(self):
        return "MinkowskiSimilarity"


class Chebyshev(SimilarityMeasure):
    """
    Chebyshev similarity
    """

    def __init__(self, coeff_1, coeff_2):
        super().__init__(coeff_1, coeff_2)
        self.d_max = 1

    def distance_measure(self):
        return np.max(np.absolute(np.subtract(self.coeff_1, self.coeff_2)))

    def __str__(self):
        return "ChebyshevSimilarity"


class Sorensen(SimilarityMeasure):
    """
    Sorensen similarity
    """

    def __init__(self, coeff_1, coeff_2):
        super().__init__(coeff_1, coeff_2)
        self.d_max = 1

    def distance_measure(self):
        nominator = np.sum(np.absolute(np.subtract(self.coeff_1, self.coeff_2)))
        denominator = np.sum(np.add(self.coeff_1, self.coeff_2))
        return nominator / denominator

    def __str__(self):
        return "SorensenSimilarity"


class Kulczynski(SimilarityMeasure):
    """
    Kulczyniski similarity
    """

    def __init__(self, coeff_1, coeff_2):
        super().__init__(coeff_1, coeff_2)
        self.d_max = np.inf

    def distance_measure(self):
        nominator = np.sum(np.absolute(np.subtract(self.coeff_1, self.coeff_2)))
        denominator = np.sum(np.minimum(self.coeff_1, self.coeff_2))
        return nominator / denominator

    def similarity_measure(self):
        coeff_dist = self.distance_measure()
        return 1 / coeff_dist

    def __str__(self):
        return "KulczynskiSimilarity"


class Lorentzian(SimilarityMeasure):
    """
    Lorentzian similarity
    """

    def __init__(self, coeff_1, coeff_2):
        super().__init__(coeff_1, coeff_2)
        self.d_max = 2 * np.log(2)

    def distance_measure(self):
        return np.sum(np.log(1 + np.absolute(np.subtract(self.coeff_1, self.coeff_2))))

    def __str__(self):
        return "LorentzianSimilarity"


class Intersection(SimilarityMeasure):
    """
    Intersection similarity
    """

    def __init__(self, coeff_1, coeff_2):
        super().__init__(coeff_1, coeff_2)
        self.d_max = 1

    def distance_measure(self):
        return np.sum(np.absolute(np.subtract(self.coeff_1, self.coeff_2))) / 2

    def __str__(self):
        return "IntersectionSimilarity"


class Czekanowski(SimilarityMeasure):
    """
    Czekanowski similarity
    """

    def __init__(self, coeff_1, coeff_2):
        super().__init__(coeff_1, coeff_2)
        self.d_max = 1

    def distance_measure(self):
        nominator = np.sum(np.absolute(np.subtract(self.coeff_1, self.coeff_2)))
        denominator = np.sum(np.add(self.coeff_1, self.coeff_2))
        return nominator / denominator

    def __str__(self):
        return "CzekanowskiSimilarity"


class Motyka(SimilarityMeasure):
    """
    Motyka similarity
    """

    def __init__(self, coeff_1, coeff_2):
        super().__init__(coeff_1, coeff_2)
        self.d_max = 1

    def distance_measure(self):
        nominator = np.sum(np.maximum(self.coeff_1, self.coeff_2))
        denominator = np.sum(np.add(self.coeff_1, self.coeff_2))
        return nominator / denominator

    def __str__(self):
        return "MotykaSimilarity"


class Ruzicka(SimilarityMeasure):
    """
    Ruzicka similarity
    """

    def similarity_measure(self):
        nominator = np.sum(np.minimum(self.coeff_1, self.coeff_2))
        denominator = np.sum(np.maximum(self.coeff_1, self.coeff_2))
        return nominator / denominator

    def __str__(self):
        return "RuzickaSimilarity"


class Tanimoto(SimilarityMeasure):
    """
    Tanimoto similarity
    """

    def __init__(self, coeff_1, coeff_2):
        super().__init__(coeff_1, coeff_2)
        self.d_max = 1

    def distance_measure(self):
        nominator = np.sum(np.subtract(np.maximum(self.coeff_1, self.coeff_2), np.minimum(self.coeff_1, self.coeff_2)))
        denominator = np.sum(np.maximum(self.coeff_1, self.coeff_2))
        return nominator / denominator

    def __str__(self):
        return "TanimotoSimilarity"


class InnerProduct(SimilarityMeasure):
    """
    Inner product similarity
    """

    def similarity_measure(self):
        return np.inner(self.coeff_1, self.coeff_2)

    def __str__(self):
        return "InnerProductSimilarity"


class HarmonicMean(SimilarityMeasure):
    """
    Harmonic Mean similarity
    """

    def similarity_measure(self):
        nominator = np.multiply(self.coeff_1, self.coeff_2)
        denominator = np.add(self.coeff_1, self.coeff_2)
        deno_no_zero_index = np.where(denominator != 0)

        return 2 * np.sum(nominator[deno_no_zero_index] / denominator[deno_no_zero_index])

    def __str__(self):
        return "HarmonicMeanSimilarity"


class Cosine(SimilarityMeasure):
    """
    Cosine similarity: in similarity_measure the default setting returns standard cosine_similarity
    """

    def similarity_measure(self):
        return smp.cosine_similarity(self.coeff_1.reshape(1, -1), self.coeff_2.reshape(1, -1))[0][0]

    def __str__(self):
        return "CosineSimilarity"


class Jaccard(SimilarityMeasure):
    """
    KumarHassebrook similarity
    """

    def distance_measure(self):
        similarity = self.similarity_measure()
        return 1 - similarity

    def similarity_measure(self):
        """
        The distance measure is the Kumar-Hassebrook similarity between two vectors
        """
        nominator = np.dot(self.coeff_1, self.coeff_2)
        denominator = np.sum(np.subtract(np.add(np.square(self.coeff_1), np.square(self.coeff_2)),
                                         np.multiply(self.coeff_1, self.coeff_2)))
        return nominator / denominator

    def __str__(self):
        return "JaccardSimilarity"


class Dice(SimilarityMeasure):
    """
    Dice similarity measure
    """

    def __init__(self, coeff_1, coeff_2):
        super().__init__(coeff_1, coeff_2)
        self.d_max = 1

    def distance_measure(self):
        similarity = self.similarity_measure()
        return 1 - similarity

    def similarity_measure(self):
        nominator = 2 * np.dot(self.coeff_1, self.coeff_2)
        denominator = np.sum(np.add(np.square(self.coeff_1), np.square(self.coeff_2)))
        return nominator / denominator

    def __str__(self):
        return "DiceSimilarity"


class Fidelity(SimilarityMeasure):
    """
    Fidelity similarity measure
    """

    def similarity_measure(self):
        return np.sum(np.sqrt(np.abs(np.multiply(self.coeff_1, self.coeff_2))))

    def __str__(self):
        return "FidelitySimilarity"


class Hellinger(SimilarityMeasure):
    """
    Hellinger similarity measure
    """

    def __init__(self, coeff_1, coeff_2):
        super().__init__(coeff_1, coeff_2)
        self.d_max = 2

    def distance_measure(self):
        inter_result = np.subtract(np.sqrt(np.abs(self.coeff_1)), np.sqrt(np.abs(self.coeff_2)))
        return np.sqrt(2 * np.sum(np.square(inter_result)))

    def __str__(self):
        return "HellingerSimilarity"


class Matusita(SimilarityMeasure):
    """
    Matusita similarity measure
    """

    def __init__(self, coeff_1, coeff_2):
        super().__init__(coeff_1, coeff_2)
        self.d_max = np.sqrt(2)

    def distance_measure(self):
        inter_result = np.subtract(np.sqrt(np.abs(self.coeff_1)), np.sqrt(np.abs(self.coeff_2)))
        return np.sqrt(np.sum(np.square(inter_result)))

    def __str__(self):
        return "MatusitaSimilarity"


class Squaredchord(SimilarityMeasure):
    """
    Squaredchord similarity measure
    """

    def __init__(self, coeff_1, coeff_2):
        super().__init__(coeff_1, coeff_2)
        self.d_max = 2

    def distance_measure(self):
        inter_result = np.subtract(np.sqrt(np.abs(self.coeff_1)), np.sqrt(np.abs(self.coeff_2)))
        return np.sum(np.square(inter_result))

    def __str__(self):
        return "SquaredchordSimilarity"


class SquaredEuclidean(SimilarityMeasure):
    """
    Class to calculate the Squared Euclidean similarity
    """

    def __init__(self, coeff_1, coeff_2):
        super().__init__(coeff_1, coeff_2)
        self.d_max = 2

    def distance_measure(self):
        return np.square(spd.euclidean(self.coeff_1, self.coeff_2))

    def __str__(self):
        return "SquaredEuclideanSimilarity"


class SquaredChiSquare(SimilarityMeasure):
    """
    Squared ChiSquare similarity measure
    """

    def __init__(self, coeff_1, coeff_2):
        super().__init__(coeff_1, coeff_2)
        self.d_max = 2

    def distance_measure(self):
        nominator = np.square(np.subtract(self.coeff_1, self.coeff_2))
        denominator = np.add(self.coeff_1, self.coeff_2)
        deno_no_zero = np.where(denominator != 0)

        return np.sum(nominator[deno_no_zero] / denominator[deno_no_zero])

    def __str__(self):
        return "SquaredChiSquare Similarity"


class ProbabilisticSymmetricChiS(SimilarityMeasure):
    """
    Squared Probabilistic Symmetric ChiSquare similarity measure
    """

    def __init__(self, coeff_1, coeff_2):
        super().__init__(coeff_1, coeff_2)
        self.d_max = 4

    def distance_measure(self):
        nominator = np.square(np.subtract(self.coeff_1, self.coeff_2))
        denominator = np.add(self.coeff_1, self.coeff_2)
        deno_no_zero_index = np.where(denominator != 0)

        return 2 * np.sum(nominator[deno_no_zero_index] / denominator[deno_no_zero_index])

    def __str__(self):
        return "Probabilistic Symmetric ChiSquare Similarity"


class Kdivergence(SimilarityMeasure):
    """
    K divergence similarity
    """

    def __init__(self, coeff_1, coeff_2):
        super().__init__(coeff_1, coeff_2)
        self.d_max = np.log(2)

    def distance_measure(self):
        denominator = np.add(self.coeff_1, self.coeff_2)
        deno_no_zero = np.where(denominator != 0)

        left_log_term = np.abs(np.divide(2 * self.coeff_1[deno_no_zero], denominator[deno_no_zero]))
        left_no_zero_index = np.where(left_log_term != 0)
        left_term = np.multiply(self.coeff_1[deno_no_zero][left_no_zero_index],
                                np.log(left_log_term)[left_no_zero_index])

        return np.sum(left_term)

    def __str__(self):
        return "Kdivergence Similarity"


class Topsoe(SimilarityMeasure):
    """
    Topsoe Similarity
    """

    def __init__(self, coeff_1, coeff_2):
        super().__init__(coeff_1, coeff_2)
        self.d_max = 2 * np.log(2)

    def distance_measure(self):
        left_log_term = np.abs(np.divide(2 * self.coeff_1, np.add(self.coeff_1, self.coeff_2)))
        left_no_zero_index = np.where(left_log_term != 0)
        left_term = np.multiply(self.coeff_1[left_no_zero_index], np.log(left_log_term[left_no_zero_index]))

        right_log_term = np.abs(np.divide(2 * self.coeff_1, np.add(self.coeff_1, self.coeff_2)))
        right_no_zero_index = np.where(right_log_term != 0)
        right_term = np.multiply(self.coeff_2[right_no_zero_index], np.log(right_log_term[right_no_zero_index]))

        left_term = np.sum(left_term)
        right_term = np.sum(right_term)
        return np.sum(np.add(left_term, right_term))

    def __str__(self):
        return "Topsoe Similarity"


class JensenShannon(SimilarityMeasure):
    """
    JensenShannon Similarity
    """

    def __init__(self, coeff_1, coeff_2):
        super().__init__(coeff_1, coeff_2)
        self.d_max = np.log(2)

    def distance_measure(self):
        left_log_term = np.abs(np.divide(2 * self.coeff_1, np.add(self.coeff_1, self.coeff_2)))
        left_no_zero_index = np.where(left_log_term != 0)
        left_term = np.multiply(self.coeff_1[left_no_zero_index], np.log(left_log_term[left_no_zero_index]))

        right_log_term = np.abs(np.divide(2 * self.coeff_1, np.add(self.coeff_1, self.coeff_2)))
        right_no_zero_index = np.where(right_log_term != 0)
        right_term = np.multiply(self.coeff_2[right_no_zero_index], np.log(right_log_term[right_no_zero_index]))

        # left_term = np.multiply(self.coeff_1, np.log(np.abs(np.divide(2 * self.coeff_1, np.add(self.coeff_1, self.coeff_2)))))
        # right_term = np.multiply(self.coeff_2, np.log(np.abs(np.divide(2 * self.coeff_2, np.add(self.coeff_1, self.coeff_2)))))

        left_term = np.sum(left_term)
        right_term = np.sum(right_term)

        return 0.5 * (left_term + right_term)

    def __str__(self):
        return "JensenShannon Similarity"


class JensenDifference(SimilarityMeasure):
    """
    Jensen Difference Similarity
    """

    def __init__(self, coeff_1, coeff_2):
        super().__init__(coeff_1, coeff_2)
        self.d_max = np.log(1.0 / 2)

    def distance_measure(self):
        left_term = np.add(np.multiply(self.coeff_1, np.log(self.coeff_1)),
                           np.multiply(self.coeff_2, np.log(self.coeff_2))) / 2
        right_term = np.multiply(np.add(self.coeff_1, self.coeff_2) / 2, np.log(np.add(self.coeff_1, self.coeff_2) / 2))
        return np.sum(np.subtract(left_term, right_term))

    def __str__(self):
        return "JensenDifference Similarity"


class AvgL1Linf(SimilarityMeasure):
    """
    Average L1 L_inf similarity
    """

    def __init__(self, coeff_1, coeff_2):
        super().__init__(coeff_1, coeff_2)
        self.d_max = 3.0 / 2

    def distance_measure(self):
        cheby_dist = spd.chebyshev(self.coeff_1, self.coeff_2)
        city_dist = spd.cityblock(self.coeff_1, self.coeff_2)
        return (cheby_dist + city_dist) / 2

    def __str__(self):
        return "Average L1 L_inf Similarity"


class VicisSymmetricChi(SimilarityMeasure):
    """
    Vicis-Symmetric Chi square similarity
    """

    def __init__(self, coeff_1, coeff_2):
        super().__init__(coeff_1, coeff_2)
        self.d_max = 1

    def distance_measure(self):
        nominator = np.square(np.subtract(self.coeff_1, self.coeff_2))
        denominator = np.maximum(self.coeff_1, self.coeff_2)
        non_zero_deno = np.where(denominator != 0)
        return np.sum(np.divide(nominator[non_zero_deno], denominator[non_zero_deno]))

    def __str__(self):
        return "VicisSymmetricChi Similarity"


class MinSymmetricChi(SimilarityMeasure):
    """
    MinSymmetric Chisquare similarity
    """

    def __init__(self, coeff_1, coeff_2):
        super().__init__(coeff_1, coeff_2)
        self.d_max = 1

    def distance_measure(self):
        coeff_1_non_zero = np.where(self.coeff_1 != 0)
        coeff_2_non_zero = np.where(self.coeff_2 != 0)

        left_term = np.sum(np.divide(np.square(np.subtract(self.coeff_1, self.coeff_2))[coeff_1_non_zero],
                                     self.coeff_1[coeff_1_non_zero]))
        right_term = np.sum(np.divide(np.square(np.subtract(self.coeff_1, self.coeff_2))[coeff_2_non_zero],
                                      self.coeff_2[coeff_2_non_zero]))
        return np.minimum(left_term, right_term)

    def __str__(self):
        return "minsymmetric Chisquare Similarity"


class PearsonCorrMeasure(SimilarityMeasure):
    """
    Pearson Correlation Measure
    """

    def similarity_measure(self):
        return pearsonr(self.coeff_1, self.coeff_2)[0]

    def __str__(self):
        return "PearsonSimilarity"
