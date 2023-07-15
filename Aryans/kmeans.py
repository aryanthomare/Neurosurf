import numpy as np
import random
from sklearn.cluster import KMeans
import csv
import matplotlib.pyplot as plt

def dist(p1,p2):
    #compute the distance between two points
    return np.sqrt(np.sum((p1-p2)**2))



def get_file(file):
    #open csv file and save rows in a list
    with open(file, 'r') as file:
        reader = csv.reader(file)
        lines = np.array(list(reader),dtype=float)
        return lines
    


def normalize(data):
    #normalize the data to have mean 0 and variance 1
    #return the normalized data and the mean and variance of the original data
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def unnormalize(data,max,min):
    #normalize the data to have mean 0 and variance 1
    #return the normalized data and the mean and variance of the original data
    return (data * (max - min)) + min



def kmeans(data, k, tol=1e-6):
    #run k means clustering on data and return the cluster centers and labels
    kmeans_model = KMeans(n_clusters=k, tol=tol,)
    # Fit into our dataset fit
    kmeans_predict = kmeans_model.fit_predict(data)
    centers = kmeans_model.cluster_centers_
    print(centers)

data = np.concatenate((get_file('Neurosurf\\Aryans\\Exported_Values\\blinks\\blinkstp9.csv'), 
                       get_file('Neurosurf\\Aryans\\Exported_Values\\normal\\normaltp9.csv')), axis=0)
kmeans(data, 3)




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

blinks = get_file('Neurosurf\\Aryans\\Exported_Values\\blinks\\blinkstp9.csv')
normal = get_file('Neurosurf\\Aryans\\Exported_Values\\normal\\normaltp9.csv')

column1 = blinks[:, 0]
column2 = blinks[:, 1]
column3 = blinks[:, 2]



ax.scatter(column1, column2, column3, c='b', marker='o')


column1 = normal[:, 0]
column2 = normal[:, 1]
column3 = normal[:, 2]



ax.scatter(column1, column2, column3, c='y', marker='o')
ax.relim()
ax.autoscale_view()
# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot')
# Show the plot
plt.show()