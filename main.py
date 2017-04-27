import grid
from data_loader import DataLoader
from recommender import Recommender
import pickle
import plot_rmse_mae
import numpy as np
import pandas

# Choose dataset
data_loader = DataLoader('Data/Patio_Lawn_and_Garden_5.json', train_rand=8,test_rand=2)
# data_loader = DataLoader('Data/Baby_5.json', train_rand=8,test_rand=2)

# Split the dataset
full_train_data = data_loader.get_full_train_data()
fold_train_data = data_loader.get_fold_train_data(5)
test_data = data_loader.get_test_data()

# generate the data set for top k recommendation
top_k_data = data_loader.get_top_k_pred_set()

# get the grid
pgrid = grid.get_grid()

# Train the model, get the results and plot
recommender = Recommender()
recommender.train(fold_train_data,pgrid)
maes, rmses, x_label, std_rmse, std_mae = recommender.get_final_results()
plot_rmse_mae.plot_line_graph([rmses, maes], ["RMSES","MAES"], "RMSE VS MAE", range(1, len(rmses)+1), x_label)

# Calculate if one algorithm is truly better than the oterh
algo_scorer_maes = np.zeros((len(maes), len(maes)))
algo_scorer_rmse = np.zeros((len(maes), len(maes)))
for i in range(0,len(maes)):
    for j in range(0, len(maes)):
        algo_scorer_maes[i][j] = int(abs(maes[i] - maes[j]) > std_mae[i] + std_mae[j])
        algo_scorer_rmse[i][j] = int(abs(rmses[i] - rmses[j]) > std_rmse[i] + std_rmse[j])


# Predict on the Testset
prediction_rmse, rmse_rmse, mae_rmse = recommender.predict_best(full_train_data, test_data, "RMSE")
prediction_mae, rmse_mae, mae_mae = recommender.predict_best(full_train_data, test_data, "MAE")

# Predict the top K
top_k_prediction_rmse = recommender.get_top_k(top_k_data, "RMSE")
top_k_prediction_mae = recommender.get_top_k(top_k_data, "MAE")

# Calculate the scores for the top k with each algorithm
precision_rmse, recall_rmse, f_rmse = recommender.precision(top_k_prediction_rmse, test_data, 10)
precision_mae, recall_mae, f_mae = recommender.precision(top_k_prediction_mae, test_data, 10)


# Print the scores

print "\n\n\n\n\nComparison of Algorithms. (If one algorithm is better than the other: 1.0 else 0.0)"
print "\nAlgo scorer maes:"
df = pandas.DataFrame(algo_scorer_maes, columns=x_label, index=x_label)
print df


print "\nAlgo scorer rmses:"
df = pandas.DataFrame(algo_scorer_rmse, columns=x_label, index=x_label)
print df


print "\nThe best parameters for RMSE with score "+ str(recommender.get_best_score("RMSE")) +": "
print recommender.get_best_algo("RMSE")
print recommender.get_best_params("RMSE")
print ""
print "The best parameters for MAE with score "+ str(recommender.get_best_score("MAE")) +": "
print recommender.get_best_algo("MAE")
print recommender.get_best_params("MAE")
print ""



print "Scores on Trainingset:"

print "RMSE Scores: "
print x_label
print rmses
print "MAE Scores: "
print x_label
print maes

print "\nBest RMSE Algorithm on Testset: RMSE: "+ str(rmse_rmse) + " MAE: "+ str(mae_rmse)
print "Best MAE Algorithmon Testset: RMSE: "+ str(rmse_mae) + " MAE: "+ str(mae_mae)

print "\n\nTop K scores"
print "\nPrecision RMSE Algorithm = \t" + str(precision_rmse)
print "Recall RMSE Algorithm = \t\t" + str(recall_rmse)
print "F1 RMSE Algorithm = \t\t\t" + str(f_rmse)

print "\nPrecision MAE Algorithm = \t" + str(precision_mae)
print "Recall MAE Algorithm = \t\t" + str(recall_mae)
print "F1 MAE Algorithm = \t\t\t" + str(f_mae)

