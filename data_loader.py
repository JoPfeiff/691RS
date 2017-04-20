import json
import pandas as pd
import surprise as sup
import os
import os.path
from random import randint
import copy


class DataLoader:

    def __init__(self, file_name, train_rand=8, test_rand=2):
        self.reader = sup.Reader(line_format='item user rating', sep=";", rating_scale=(1, 5))
        self.file_name = file_name
        self.train_rand = train_rand
        self.test_rand = test_rand
        self.new_file_name_train, self.new_file_name_test =  self.load_data()
        self.train_data = self.generate_train_data()
        self.test_data = self.generate_test_data_tuple()

    def get_train_file(self):
        return self.new_file_name_test

    def generate_train_data(self):
        return sup.Dataset.load_from_file(self.new_file_name_train, reader=self.reader)

    def get_full_train_data(self):
        data = copy.deepcopy(self.train_data)
        return data.build_full_trainset()

    def get_fold_train_data(self,k):
        data = copy.deepcopy(self.train_data)
        data.split(n_folds=k)
        return data

    def get_test_data(self):
        return self.test_data

    def load_data(self):

        data = {}
        new_file_name = self.file_name.split(".")
        new_file_name_train = new_file_name[0] + "_train_" + str(self.train_rand) + ".csv"
        new_file_name_test = new_file_name[0] + "_test_" + str(self.test_rand) + ".csv"

        if os.path.isfile(new_file_name_train) and os.path.isfile(new_file_name_test):
            return os.path.expanduser(new_file_name_train), os.path.expanduser(new_file_name_test)

        else:
            with open(self.file_name) as data_file:
                id = 0
                for line in data_file:
                    data[id] = json.loads(line)
                    id += 1

            data_csv_string_train = ""
            data_csv_string_test = ""

            for elem in data.iteritems():
                if randint(1, self.train_rand + self.test_rand) <= self.train_rand:
                    data_csv_string_train += elem[1]['asin'] + ";" + elem[1]['reviewerID'] + ";" + str(
                        elem[1]['overall']) + "\n"
                else:
                    data_csv_string_test += elem[1]['asin'] + ";" + elem[1]['reviewerID'] + ";" + str(
                        elem[1]['overall']) + "\n"

            with open(new_file_name_train, "w") as text_file:
                text_file.write(data_csv_string_train)

            with open(new_file_name_test, "w") as text_file:
                text_file.write(data_csv_string_test)

            return os.path.expanduser(new_file_name_train), os.path.expanduser(new_file_name_test)

    # def generate_test_data(file):
    #
    #     reviewerID = []
    #     asin = []
    #     score = []
    #
    #     with open(file) as f:
    #         for line in f:
    #             line = line.split(";")
    #             asin.append(line[1])
    #             reviewerID.append(line[0])
    #             score.append(float(line[2]))
    #
    #     return reviewerID, asin, score

    def generate_test_data_tuple(self):

        test_data = []
        with open(self.new_file_name_test) as f:
            for line in f:
                line = line.split(";")
                test_data.append((line[1], line[0], float(line[2])))
        return test_data

    def generate_user_item_dict(self):
        user_item_dict = {}
        with open(self.new_file_name_train) as f:
            for line in f:
                line = line.split(";")
                if line[1] not in user_item_dict:
                    user_item_dict[line[1]] = {}
                user_item_dict[line[1]][line[0]] = line[2]
        return user_item_dict

    def get_unique_item_user_list(self):
        unique_item_list = []
        unique_user_list = []
        with open(self.new_file_name_train) as f:
            for line in f:
                line = line.split(";")
                if line[0] not in unique_item_list:
                    unique_item_list.append(line[0])
                if line[1] not in unique_user_list:
                    unique_user_list.append(line[1])
        return unique_item_list, unique_user_list

    def get_top_k_pred_set(self):

        user_item_dict = self.generate_user_item_dict()
        unique_item_list, unique_user_list = self.get_unique_item_user_list()
        top_k_set = []

        for user in unique_user_list:
            for item in unique_item_list:
                if item not in user_item_dict[user]:
                    top_k_set.append((user, item, 0))
        return top_k_set  # reviewerID, asin




