
import grid
from data_loader import DataLoader
from recommender import Recommender
import pickle

# data_loader = DataLoader('Data/Patio_Lawn_and_Garden_5.json', train_rand=8,test_rand=2)
data_loader = DataLoader('Data/Baby_5.json', train_rand=8,test_rand=2)

full_train_data = data_loader.get_full_train_data()
fold_train_data = data_loader.get_fold_train_data(5)
test_data = data_loader.get_test_data()
top_k_data = data_loader.get_top_k_pred_set()

pgrid = grid.get_grid()
recommender = Recommender()

recommender.train(fold_train_data,pgrid)

print "The best parameters for RMSE with score "+ str(recommender.get_best_score("RMSE")) +": "
print recommender.get_best_algo("RMSE")
print recommender.get_best_params("RMSE")
print ""
print "The best parameters for MAE with score "+ str(recommender.get_best_score("MAE")) +": "
print recommender.get_best_algo("MAE")
print recommender.get_best_params("MAE")
print ""


prediction, rmse, mae = recommender.predict_best(full_train_data, test_data, "RMSE")

top_k_prediction = recommender.get_top_k(top_k_data, "RMSE")

with open('pick_baby.pickle', 'wb') as handle:
    pickle.dump(top_k_prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('pick.pickle', 'rb') as handle:
#     top_k_prediction = pickle.load(handle)

precision, recall, f = recommender.precision(top_k_prediction, test_data, 10)

print "precision = " + str(precision)
print "recall = " + str(recall)
print "f = " + str(f)
print "done"





