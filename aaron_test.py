import pandas as pd
import numpy as np
import surprise as sup
from surprise import SVD, SlopeOne, KNNBasic, KNNWithMeans, NMF, CoClustering
import os
import os.path
from surprise import Dataset
from surprise import evaluate, print_perf
import matplotlib.pyplot as plt


import jonas_test

#building off of what jonas has done

print "program start"

train_file, test_file = jonas_test.load_data('Data/Patio_Lawn_and_Garden_5.json', 9,1)

reader = sup.Reader(line_format = 'item user rating', sep= ";", rating_scale=(1,5), )

data = sup.Dataset.load_from_file(train_file, reader=reader)
data.split(n_folds=5)

#aggregate mean absolute errors
maes = []
#aggregate rmses
rmses = []

for algorithm in [SVD(), KNNBasic(), KNNWithMeans(), NMF(), CoClustering()]:
	perf = evaluate(algorithm, data, measures=['RMSE', 'MAE']) #evaluate on these errors
	print_perf(perf)
	
	maes.append(perf["mae"]) #append these to their respective lists
	rmses.append(perf["rmse"])
	
#chart MAE and RMSE across algorithms
finalmaes = [alg_scores[-1] for alg_scores in maes] #grab the last element which is the average score
finalrmses = [alg_scores[-1] for alg_scores in rmses] #grab the last element which is the average score

inds = np.arange(len(finalmaes))
labels = ["SVD", "KNNBasic", "KNNWithMeans", "NMF", "CoClustering"]


plt.figure(2, figsize=(6,4))  #6x4 is the aspect ratio for the plot
plt.plot(inds,values2,'or-', linewidth=3) #Plot the first series in red with circle marker
plt.plot(inds,values,'sb-', linewidth=3)


plt.grid(True) #Turn the grid on
plt.ylabel("Error") #Y-axis label
plt.xlabel("Model") #X-axis label
plt.title("Advertising Model Selection: \nError vs Value on Decision Tree and \nRandom Forest Classifiers") #Plot title
plt.ylim(0,2) #Set yaxis range
plt.legend(["MAE", "RMSE"],loc="best")


#Make sure labels and titles are inside plot area
plt.tight_layout()

#Save the chart
plt.savefig("../Figures/example_line_plot2.pdf")

#Displays the plots.
#You must close the plot window for the code following each show()
#to continue to run
plt.show()