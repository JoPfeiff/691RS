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
import grid
import plot_rmse_mae
import load_train_test_data

from operator import itemgetter


def get_rmse_mae(algo, trainset, testset):
    # algo.train(trainset)
    predictions = algo.test(testset)
    # print predictions
    return sup.accuracy.rmse(predictions, verbose=True), sup.accuracy.mae(predictions, verbose=True)

def prediction(algo, predset):
    algo.train(trainset)
    prediction_set = []
    for i in range(0, len(reviewerID)):
        prediction_set.append(algo.predict(uid=predset[0][i], iid=predset[1][i]))
    return prediction_set

def get_top_k(algo, top_k_set, k=10):
    print "starting prediction of topK"
    predictions = algo.test(top_k_set)
    print "topK predicted"
    top_k_dict = {}
    print "build topK dict"
    for elem in predictions:
        if elem.uid not in top_k_dict:
            top_k_dict[elem.uid] = []
        top_k_dict[elem.uid].append([elem.iid, elem.est])
    print "topK dict built \n Sorting dict"
    for user, item_rating in top_k_dict.iteritems():
        list  = sorted(item_rating, key=itemgetter(1), reverse=True)
        new_list = []
        for i in range(0,k):
            new_list.append(list[i])
        top_k_dict[user] = new_list
        # for i in range(k, len(item_rating)):
        #     item_rating.remove(k)

    return top_k_dict




train_file, test_file = load_train_test_data.load_data('Data/Patio_Lawn_and_Garden_5.json', train_rand=8,test_rand=2)

reviewerID, asin, score = load_train_test_data.generate_test_data(test_file)


reader = sup.Reader(line_format = 'item user rating', sep= ";", rating_scale=(1,5) )

#
data = sup.Dataset.load_from_file(train_file, reader=reader)
trainset = data.build_full_trainset()
data.split(n_folds=5)
testset = load_train_test_data.generate_test_data_tuple(test_file)

user_item_dict = load_train_test_data.generate_user_item_dict(train_file)
unique_item_list, unique_user_list= load_train_test_data.get_unique_item_user_list(train_file)
top_k_pred_set = load_train_test_data.get_top_k_pred_set(user_item_dict, unique_item_list, unique_user_list)

algo = KNNBasic()
algo.train(trainset)
pred = get_top_k(algo, top_k_pred_set, k=10)


# aggregate mean absolute errors
maes = []
# aggregate rmses
rmses = []

algo_param_scores ={}

best_MAE = 999
best_MAE_algo_params = None
best_MAE_algo = None
best_RMSE = 999
best_RMSE_algo_params = None
best_RMSE_algo = None



algo_dict = grid.get_grid()

for algo, params in algo_dict.iteritems():
    grid_search = GridSearch(algo, params, measures=['RMSE', 'MAE'])
    grid_search.evaluate(data)

    test = grid_search.cv_results
    # I don't know why but sumetimes cv_results RMSE is empty? This is why Im doing it like this:
    for i in range(0, len(grid_search.cv_results['scores'])):
        maes.append(grid_search.cv_results['scores'][i]['MAE'])
        rmses.append(grid_search.cv_results['scores'][i]['RMSE'])

    # Get best overall algorithm for MAE and RMSE respectively
    if grid_search.best_score["mae"] < best_MAE:
        best_MAE = grid_search.best_score["mae"]
        best_MAE_algo_params = grid_search.best_params['MAE']
        best_MAE_algo = grid_search.best_estimator['MAE']
    if grid_search.best_score["rmse"] < best_RMSE:
        best_RMSE = grid_search.best_score["rmse"]
        best_RMSE_algo_params = grid_search.best_params["rmse"]
        best_RMSE_algo = grid_search.best_estimator["rmse"]

    # get best params of each algorithm
    algo_param_scores[grid_search.best_estimator["rmse"]] = grid_search.best_score["rmse"]
    algo_param_scores[grid_search.best_estimator["mae"]] = grid_search.best_score["mae"]

    # best  = grid_search.best_estimator["rmse"]
    # best.train(trainset)


best_RMSE_algo.train(trainset)
get_top_k(best_RMSE_algo, top_k_pred_set)

for algo, s in algo_param_scores.iteritems():
    algo.train(trainset)
    rmse, mae = get_rmse_mae(algo, trainset, testset)
    get_top_k(algo, top_k_pred_set)



plot_rmse_mae.plot_line_graph([rmses,maes], ["RMSES","MAES"], "RMSE VS MAE", range(1, len(rmses)+1))




