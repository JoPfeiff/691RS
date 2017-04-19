import json
import pandas as pd
import numpy as np
import surprise as sup
from surprise import SVD
import os
import os.path
from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf



def load_data(file_name):
    data = {}
    # file_name = 'Data/Patio_Lawn_and_Garden_5.json'
    new_file_name = file_name.split(".")
    new_file_name = new_file_name[0] + ".csv"

    if os.path.isfile(new_file_name):
        return os.path.expanduser(new_file_name)

    else:
        with open(file_name) as data_file:
            id = 0
            for line in data_file:
                data[id] = json.loads(line)
                id+=1

        data_csv_string = ""

        for elem in data.iteritems():
            data_csv_string += elem[1]['reviewerID'] + ";" + elem[1]['asin'] + ";" + str(elem[1]['overall']) + "\n"

        with open(new_file_name, "w") as text_file:
            text_file.write(data_csv_string)

        return os.path.expanduser(new_file_name)

file = load_data('Data/Patio_Lawn_and_Garden_5.json')

reader = sup.Reader(line_format = 'item user rating', sep= ";", rating_scale=(1,5), )

#
data = sup.Dataset.load_from_file(file, reader=reader)
data.split(n_folds=5)

# We'll use the famous SVD algorithm.
algo = SVD()

# Evaluate performances of our algorithm on the dataset.
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

print_perf(perf)



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




