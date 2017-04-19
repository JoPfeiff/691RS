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


train_file, test_file = load_train_test_data.load_data('Data/Patio_Lawn_and_Garden_5.json', train_rand=8,test_rand=2)

reviewerID, asin, score = load_train_test_data.generate_test_data(test_file)


reader = sup.Reader(line_format = 'item user rating', sep= ";", rating_scale=(1,5) )

#
data = sup.Dataset.load_from_file(train_file, reader=reader)
trainset = data.build_full_trainset()
data.split(n_folds=5)



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

    if grid_search.best_score["mae"] < best_MAE:
        best_MAE = grid_search.best_score["mae"]
        best_MAE_algo_params = grid_search.best_params['MAE']
        best_MAE_algo = grid_search.best_estimator['MAE']
    if grid_search.best_score["rmse"] < best_RMSE:
        best_RMSE = grid_search.best_score["rmse"]
        best_RMSE_algo_params = grid_search.best_params["rmse"]
        best_RMSE_algo = grid_search.best_estimator["rmse"]

    # algo_param_scores[grid_search.best_estimator] = {"RMSE":grid_search.best_score["rmse"], "MAE": grid_search.best_score["mae"]}

    best  = grid_search.best_estimator["rmse"]
    best.train(trainset)
    for i in range(0, len(reviewerID)):
        prediction = best.predict(uid=reviewerID[i], iid=asin[i], r_ui=score[i])
        if prediction.est < 4.18 or prediction.est > 4.2:
            print(prediction.est)


    print "done"

# for algo, scores in algo_param_scores.iteritems():
#     algo.predict

print "best MAE: " + str(best_MAE)
print " algo :"
print best_MAE_algo
print "best RMSE: " + str(best_RMSE)
print " algo :"
print best_RMSE_algo


plot_rmse_mae.plot_line_graph([rmses,maes], ["RMSES","MAES"], "RMSE VS MAE", range(1, len(rmses)+1))


# pdata = pd.from_dict(data)
# pdata = pd.DataFrame.from_dict(data, orient='index')
#
# users = pd.Series.unique(pdata['reviewerID'])
# products = pd.Series.unique(pdata['asin'])
#
# colmat = pd.DataFrame(np.zeros((len(users),len(products))), columns=products, index=users  )


# for r in pdata.iterrows():
#     score = r[1]['overall']
#     reviewer = r[1]['reviewerID']
#     product = r[1]['asin']
#
#     colmat[[product]].loc[reviewer] = score




