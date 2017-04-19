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

def load_data(file_name, train_rand=8, test_rand=2):

    data = {}
    # file_name = 'Data/Patio_Lawn_and_Garden_5.json'
    new_file_name = file_name.split(".")
    new_file_name_train = new_file_name[0] +"_train_"+ str(train_rand)+ ".csv"
    new_file_name_test = new_file_name[0] +"_test_"+ str(test_rand)+ ".csv"

    if os.path.isfile(new_file_name_train) and os.path.isfile(new_file_name_test):
        return os.path.expanduser(new_file_name_train), os.path.isfile(new_file_name_test)

    else:
        with open(file_name) as data_file:
            id = 0
            for line in data_file:
                data[id] = json.loads(line)
                id+=1

        data_csv_string_train = ""
        data_csv_string_test = ""

        for elem in data.iteritems():
            if randint(1, train_rand+test_rand) <= train_rand:
                data_csv_string_train += elem[1]['reviewerID'] + ";" + elem[1]['asin'] + ";" + str(elem[1]['overall']) + "\n"
            else:
                data_csv_string_test += elem[1]['reviewerID'] + ";" + elem[1]['asin'] + ";" + str(
                    elem[1]['overall']) + "\n"

        with open(new_file_name_train, "w") as text_file:
            text_file.write(data_csv_string_train)

        with open(new_file_name_test, "w") as text_file:
            text_file.write(data_csv_string_test)

        return os.path.expanduser(new_file_name_train), os.path.expanduser(new_file_name_train)

train_file, test_file = load_data('Data/Patio_Lawn_and_Garden_5.json', train_rand=8,test_rand=2)

reader = sup.Reader(line_format = 'item user rating', sep= ";", rating_scale=(1,5) )

#
data = sup.Dataset.load_from_file(train_file, reader=reader)
data.split(n_folds=5)

# aggregate mean absolute errors
maes = []
# aggregate rmses
rmses = []

best_MAE = 999
best_MAE_algo = None
best_RMSE = 999
best_RMSE_algo = None

algo_dict = grid.get_grid()

for algo, params in algo_dict.iteritems():
    grid_search = GridSearch(algo, params, measures=['RMSE', 'MAE'])
    grid_search.evaluate(data)
    if grid_search.best_score["mae"] < best_MAE:
        best_MAE = grid_search.best_score["mae"]
        best_MAE_algo = grid_search.best_params
    if grid_search.best_score["rmse"] < best_RMSE:
        best_RMSE = grid_search.best_score["rmse"]
        best_RMSE_algo = grid_search.best_params

print "best MAE: " + str(best_MAE)
print " algo :"
print best_MAE_algo
print "best RMSE: " + str(best_RMSE)
print " algo :"
print best_RMSE_algo





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




