import surprise as sup
from surprise import GridSearch
import numpy as np
from operator import itemgetter


class Recommender:
    def __init__(self):
        self.train_data = None
        self.grid = None
        self.maes = []
        self.rmses = []
        self.names = []
        self.algo_param_scores = {}


        self.best_MAE = [999,None,None]
        self.best_RMSE = [999,None,None]

        self.top_k_dict = None

    def train(self, train_data, grid):
        self.train_data = train_data
        self.grid = grid

        for algo, params in grid.iteritems():
            grid_search = GridSearch(algo, params, measures=['RMSE', 'MAE'])
            grid_search.evaluate(train_data)

            test = grid_search.cv_results
            # I don't know why but sumetimes cv_results RMSE is empty? This is why Im doing it like this:
            '''for i in range(0, len(grid_search.cv_results['scores'])):
                self.maes.append(grid_search.cv_results['scores'][i]['MAE'])
                self.rmses.append(grid_search.cv_results['scores'][i]['RMSE'])'''

            # Get best overall algorithm for MAE and RMSE respectively
            if grid_search.best_score["mae"] < self.best_MAE[0]:
                self.best_MAE[0] = grid_search.best_score["mae"]
                self.best_MAE[1] = grid_search.best_params['MAE']
                self.best_MAE[2] = grid_search.best_estimator['MAE']
            if grid_search.best_score["rmse"] < self.best_RMSE[0]:
                self.best_RMSE[0] = grid_search.best_score["rmse"]
                self.best_RMSE[1] = grid_search.best_params["rmse"]
                self.best_RMSE[2] = grid_search.best_estimator["rmse"]

            # get best params of each algorithm
			
            self.algo_param_scores[grid_search.best_estimator["rmse"]] = grid_search.best_score["rmse"]
            self.algo_param_scores[grid_search.best_estimator["mae"]] = grid_search.best_score["mae"]
            self.maes.append(grid_search.best_score["mae"])
            self.rmses.append(grid_search.best_score["rmse"])
            self.names.append(str(algo).split(" ")[0].split(".")[-1])

            # best  = grid_search.best_estimator["rmse"]
            # best.train(trainset)
        print self.maes
        print self.rmses
        print self.names

    def predict_best(self,train_data, test_data, scorer):
        if scorer in ["MAE", "mae"]:
            algo = self.best_MAE[2]
        elif scorer in ["RMSE", "rmse"]:
            algo = self.best_RMSE[2]
        else:
            print "scorer not trained"
            return False
        algo.train(train_data)
        predictions = algo.test(test_data)
        rmse = sup.accuracy.rmse(predictions, verbose=True)
        mae = sup.accuracy.mae(predictions, verbose=True)

        return predictions, rmse, mae

    def get_best_params(self, scorer):
        if scorer in ["MAE", "mae"]:
            return self.best_MAE[1]
        elif scorer in ["RMSE", "rmse"]:
            return self.best_RMSE[1]
        else:
            print "scorer not trained"
            return False


    def get_best_algo(self, scorer):
        if scorer in ["MAE", "mae"]:
            return self.best_MAE[2]
        elif scorer in ["RMSE", "rmse"]:
            return self.best_RMSE[2]
        else:
            print "scorer not trained"
            return False

    def get_best_score(self, scorer):
        if scorer in ["MAE", "mae"]:
            return self.best_MAE[0]
        elif scorer in ["RMSE", "rmse"]:
            return self.best_RMSE[0]
        else:
            print "scorer not trained"
            return False

    def get_top_k(self, top_k_set, scorer, k=10):
        if scorer in ["MAE", "mae"]:
            algo = self.best_MAE[2]
        elif scorer in ["RMSE", "rmse"]:
            algo = self.best_RMSE[2]
        else:
            print "scorer not trained"
            return False

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
            list = sorted(item_rating, key=itemgetter(1), reverse=True)
            new_list = []
            for i in range(0, k):
                new_list.append(list[i])
            top_k_dict[user] = new_list
            # for i in range(k, len(item_rating)):
            #     item_rating.remove(k)
        self.top_k_dict = top_k_dict
        return top_k_dict

    def get_final_results():
        return self.maes, self.rmses, self.names

    def precision(self, top_k_dict, test_data,k):
        user_item_dict ={}
        for line in test_data:
            if line[0] not in user_item_dict:
                user_item_dict[line[0]] = {}
            user_item_dict[line[0]][line[1]] = line[2]


        precision_list = []
        recall_list = []
        item_counter= []
        for user, items in top_k_dict.iteritems():
            counter = 0.0
            if user not in user_item_dict:
                # precision_list.append(0.0)
                # precision_list.append(0.0)
                # recall_list.append(0.0)
                # item_counter.append(0.0)
                ""
            else:
                for item in items:
                    if item[0] in user_item_dict[user]:
                        counter += 1
                precision_list.append(counter/len(user_item_dict[user]))
                recall_list.append(counter/k)
                item_counter.append(len(user_item_dict[user]))
        maxi = np.max(item_counter)

        average_nr_items = np.average(item_counter)
        precision = np.average(precision_list)
        recall = np.average(recall_list)
        f = (2*precision*recall)/(precision+recall)
        return precision, recall, f



    #def predict_best_algo(self, test_data, scorer):

