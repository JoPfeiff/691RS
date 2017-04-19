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


def load_data(file_name, train_rand=8, test_rand=2):

    data = {}
    # file_name = 'Data/Patio_Lawn_and_Garden_5.json'
    new_file_name = file_name.split(".")
    new_file_name_train = new_file_name[0] +"_train_"+ str(train_rand)+ ".csv"
    new_file_name_test = new_file_name[0] +"_test_"+ str(test_rand)+ ".csv"

    if os.path.isfile(new_file_name_train) and os.path.isfile(new_file_name_test):
        return os.path.expanduser(new_file_name_train),os.path.expanduser(new_file_name_test)

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
                data_csv_string_train += elem[1]['asin'] + ";" +elem[1]['reviewerID'] + ";" +  str(elem[1]['overall']) + "\n"
            else:
                data_csv_string_test += elem[1]['asin'] + ";" +elem[1]['reviewerID'] + ";" +  str(elem[1]['overall']) + "\n"

        with open(new_file_name_train, "w") as text_file:
            text_file.write(data_csv_string_train)

        with open(new_file_name_test, "w") as text_file:
            text_file.write(data_csv_string_test)

        return os.path.expanduser(new_file_name_train), os.path.expanduser(new_file_name_train)


def generate_test_data(file):

    reviewerID = []
    asin = []
    score = []

    with open(file) as f:
        for line in f:
            line = line.split(";")
            asin.append(line[0])
            reviewerID.append(line[1])
            score.append(float(line[2]))

    return reviewerID, asin, score