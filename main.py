
import grid
from data_loader import DataLoader
from recommender import Recommender


data_loader = DataLoader('Data/Patio_Lawn_and_Garden_5.json', train_rand=8,test_rand=2)

full_train_data = data_loader.get_full_train_data()
fold_train_data = data_loader.get_fold_train_data(5)
test_data = data_loader.get_test_data()
top_k_data = data_loader.get_top_k_pred_set()

pgrid = grid.get_grid()
recommender = Recommender()

recommender.train(fold_train_data,pgrid)
prediction, rmse, mae = recommender.predict_best(full_train_data, test_data, "RMSE")
top_k_prediction = recommender.get_top_k(top_k_data, "RMSE")

print "done"
