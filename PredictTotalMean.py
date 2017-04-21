from surprise import AlgoBase
from surprise import Dataset
from surprise import evaluate
import numpy as np

class PredictTotalMean(AlgoBase):

    def __init__(self):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)

    def train(self, trainset):

        AlgoBase.train(self, trainset)

        # Compute the average rating.
        self.the_mean = np.mean([r for (_, _, r) in
                                 self.trainset.all_ratings()])

    def estimate(self, u, i):

        return self.the_mean
