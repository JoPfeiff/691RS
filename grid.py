import json
import pandas as pd
import numpy as np
import surprise as sup
from surprise import SVD
import os
import os.path
from surprise import SVD, SlopeOne, KNNBasic, KNNWithMeans, NMF, CoClustering
from surprise import Dataset, GridSearch
from surprise import evaluate, print_perf, similarities
from random import randint



def get_grid():
    similarity_dict = [
                        # {'name': 'cosine',
                        # 'user_based': False
                        # },
                       {'name': 'cosine',
                        'user_based': True
                        },
                       # {'name': 'pearson',
                       #  'user_based': False
                       #  },
                       # {'name': 'pearson',
                       #  'user_based': True
                       #  },
                       # {'name': 'pearson_baseline',
                       #  'user_based': False
                       #  },
                       # {'name': 'pearson_baseline',
                       #  'user_based': True
                       #  }
                       ]

    knnbasic_param_grid = {
                          'k': range(20, 50, 5),
                           # 'min_k': (1, 20, 5),
                           'sim_options': similarity_dict
                           }

    svd_param_grid = {
                      'n_epochs': range(1, 10, 2),
                      # 'lr_all': np.arange(0.001, 0.01, 0.003),
                      # 'biased': [True, False],
                      # 'reg_all': np.arange(0.1, 0.8, 0.2),
                      }

    knnmeans_param_grid = {
                           "k": range(10, 50, 5),
                           'sim_options': similarity_dict
                      }

    nmf_param_grid = {
                        "n_factors": [5, 10, 15, 20, 30, 50]
						
                     }

    coclust_param_grid = {
                        "n_cltr_u": [3, 5, 10],
                        "n_cltr_i": [3, 5, 10]
                         }



    algo_dict = {
                    KNNBasic: knnbasic_param_grid,
                    SVD: svd_param_grid,
                    # KNNWithMeans: knnmeans_param_grid,
                    # NMF: nmf_param_grid,
                    # CoClustering: coclust_param_grid,
                }

    return algo_dict
