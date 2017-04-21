
import grid
from data_loader import DataLoader
from recommender import Recommender
import pickle
import plot_rmse_mae

data_loader = DataLoader('Data/Patio_Lawn_and_Garden_5.json', train_rand=8,test_rand=2)
# data_loader = DataLoader('Data/Baby_5.json', train_rand=8,test_rand=2)

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

maes, rmses, x_label = recommender.get_final_results()
plot_rmse_mae.plot_line_graph([rmses, maes], ["RMSES","MAES"], "RMSE VS MAE", range(1, len(rmses)+1), x_label)

print "X Labels: "
print x_label
print "RMSE Scores: "
print rmses
print "MAE Scores: "
print maes


prediction_rmse, rmse_rmse, mae_rmse = recommender.predict_best(full_train_data, test_data, "RMSE")
prediction_mae, rmse_mae, mae_mae = recommender.predict_best(full_train_data, test_data, "MAE")

print "Best RMSE Algorithm: RMSE: "+ str(rmse_rmse) + " MAE: "+ str(mae_rmse)
print "Best MAE Algorithm: RMSE: "+ str(rmse_mae) + " MAE: "+ str(mae_mae)

top_k_prediction_rmse = recommender.get_top_k(top_k_data, "RMSE")

with open('pick_rmse.pickle', 'wb') as handle:
    pickle.dump(top_k_prediction_rmse, handle, protocol=pickle.HIGHEST_PROTOCOL)

top_k_prediction_mae = recommender.get_top_k(top_k_data, "MAE")

with open('pick_mae.pickle', 'wb') as handle:
    pickle.dump(top_k_prediction_mae, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('pick.pickle', 'rb') as handle:
#     top_k_prediction = pickle.load(handle)

precision_rmse, recall_rmse, f_rmse = recommender.precision(top_k_prediction_rmse, test_data, 10)

print "precision _rmse= " + str(precision_rmse)
print "recall _rmse= " + str(recall_rmse)
print "f _rmse= " + str(f_rmse)

precision_mae, recall_mae, f_mae = recommender.precision(top_k_prediction_mae, test_data, 10)

print "precision _mae= " + str(precision_mae)
print "recall _mae= " + str(recall_mae)
print "f _mae= " + str(f_mae)
print "done"











print "X Labels: "
print x_label
print "RMSE Scores: "
print rmses
print "MAE Scores: "
print maes

print "Best RMSE Algorithm: RMSE: "+ str(rmse_rmse) + " MAE: "+ str(mae_rmse)
print "Best MAE Algorithm: RMSE: "+ str(rmse_mae) + " MAE: "+ str(mae_mae)


print "precision _rmse= " + str(precision_rmse)
print "recall _rmse= " + str(recall_rmse)
print "f _rmse= " + str(f_rmse)

print "precision _mae= " + str(precision_mae)
print "recall _mae= " + str(recall_mae)
print "f _mae= " + str(f_mae)

