import surprise as sup
from surprise import SVD, SlopeOne, KNNBasic, KNNWithMeans, NMF, CoClustering
from surprise import Dataset, GridSearch
import numpy as np
from PredictTotalMean import PredictTotalMean

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
                       {'name': 'pearson',
                        'user_based': True
                        },
                       # {'name': 'pearson_baseline',
                       #  'user_based': False
                       #  },
                       # {'name': 'pearson_baseline',
                       #  'user_based': True
                       #  }
                       ]

    knnbasic_param_grid = {
                          'k': range(43, 45, 1),
                           'min_k': (1, 6, 1),
                           'sim_options': similarity_dict
                           }
    #Best {'sim_options': {'name': 'cosine', 'user_based': True}, 'k': 42, 'min_k': 2}
    #Best {'sim_options': {'name': 'cosine', 'user_based': True}, 'k': 42, 'min_k': 1}




    # svd_param_grid = {
    #                   'n_epochs': [1,5,10],
    #                   'lr_all': np.arange(.0000, 0.002, .0001),# .003, .005, .007, .01, .05]
    #                   # 'biased': [True, False],
    #                   'reg_all': np.arange(0.1, 0.8, 0.2),
    #                   }

    # svd_param_grid = {
    #                     'n_epochs': range(8, 12, 1),
    #                     'lr_all': np.arange(0.002, 0.019, 0.002),
    #                     'biased': [True],
    #                     'reg_all': np.arange(0.1, 0.9, 0.1),
    #                   }
    svd_param_grid = {
                        'n_epochs': [10],
                        'lr_all': [0.0045,0.018],
                        'biased': [True],
                        'reg_all': [0.45,0.0]
                      }
    # Best:
    # {'lr_all': 0.017999999999999999, 'reg_all': 0.45000000000000001, 'biased': True, 'n_epochs': 10}
    # {'lr_all': 0.0045000000000000005, 'reg_all': 0.0, 'biased': True, 'n_epochs': 10}



    knnmeans_param_grid =     {
                                'k': [45],
                                'min_k': [1,15],
                                'sim_options': similarity_dict
                            }

    # best {'sim_options': {'name': 'cosine', 'user_based': True}, 'k': 45, 'min_k': 15}
    # best {'sim_options': {'name': 'cosine', 'user_based': True}, 'k': 45, 'min_k': 1}



    nmf_param_grid = {
                        "n_factors": [ 20,22],
                        "reg_pu": np.arange(0.008,0.0081,0.001),
                        "reg_qi": np.arange(0.008,0.0081,0.001),
                        "reg_bu": [0.0018,0.0012]#np.arange(0.0003,0.002,0.0003),
                     }
    #BEST {'n_factors': 20, 'reg_bu': 0.0017999999999999997, 'reg_qi': 0.0080000000000000002, 'reg_pu': 0.0080000000000000002}
    #BEST {'n_factors': 22, 'reg_bu': 0.0011999999999999999, 'reg_qi': 0.0080000000000000002, 'reg_pu': 0.0080000000000000002}


    coclust_param_grid = {
                        "n_cltr_u": range(1,2,1),
                        "n_cltr_i": range(1,4,1),
                        "n_epochs": [1,5]
                         }
    # {'n_cltr_u': 1, 'n_cltr_i': 1, 'n_epochs': 1}
    # best {'n_cltr_u': 1, 'n_cltr_i': 3, 'n_epochs': 5}



    algo_dict = {
                    PredictTotalMean: {},                 #ok
                    KNNBasic: knnbasic_param_grid,          #ok
                    SVD: svd_param_grid,                  # ok
                    KNNWithMeans: knnmeans_param_grid,      #ok
                    NMF: nmf_param_grid,                  # ok
                    CoClustering: coclust_param_grid,
                    SlopeOne:{}                           # ok
                }

    return algo_dict
