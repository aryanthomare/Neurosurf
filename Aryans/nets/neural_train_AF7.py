# import tensorflow.keras as keras
# import tensorflow as tf
from numpy import genfromtxt
import os
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib




blink_AF7 = genfromtxt('Neurosurf\\Aryans\\Exported_Values\\blinks\\blinksAF7.csv', delimiter=',')
blink_AF8 = genfromtxt('Neurosurf\\Aryans\\Exported_Values\\blinks\\blinksAF8.csv', delimiter=',')
blink_TP9 = genfromtxt('Neurosurf\\Aryans\\Exported_Values\\blinks\\blinksTP9.csv', delimiter=',')
blink_TP10 = genfromtxt('Neurosurf\\Aryans\\Exported_Values\\blinks\\blinksTP10.csv', delimiter=',')

normal_AF7= genfromtxt('Neurosurf\\Aryans\\Exported_Values\\normal\\normalAF7.csv', delimiter=',')
normal_AF8= genfromtxt('Neurosurf\\Aryans\\Exported_Values\\normal\\normalAF8.csv', delimiter=',')
normal_TP9= genfromtxt('Neurosurf\\Aryans\\Exported_Values\\normal\\normalTP9.csv', delimiter=',')
normal_TP10= genfromtxt('Neurosurf\\Aryans\\Exported_Values\\normal\\normalTP10.csv', delimiter=',')



blinks_df_AF7 = pd.DataFrame(blink_AF7,columns=['alpha_AF7','beta_AF7','delta_AF7','theta_AF7','gamma_AF7'])


normal_df_AF7 = pd.DataFrame(normal_AF7,columns=['alpha_AF7','beta_AF7','delta_AF7','theta_AF7','gamma_AF7'])


blinks_df_AF7['blink']=1



normal_df_AF7['blink']=0

df = pd.concat([blinks_df_AF7, normal_df_AF7], axis=0)

data = shuffle(df)

print(data)



train_df, test_df = train_test_split(data, test_size=0.2)
print(train_df.shape, test_df.shape)


train_X = train_df[train_df.columns.difference(["blink"])]
train_y = train_df["blink"]
test_X = train_df[train_df.columns.difference(["blink"])]
test_y = train_df["blink"]

clf = svm.NuSVC(gamma="auto")
clf.fit(train_X.values, train_y.values)


predicted = clf.predict(test_X)
#print(predicted)
print(accuracy_score(test_y, predicted))

joblib.dump(clf, "model_af7.pkl") 


# selected_features = ['alpha_AF7','_AF7']  # replace with the features you want to visualize

# # Extract the two features from the dataset for training and testing
# train_X_vis = train_df[selected_features]
# test_X_vis = test_df[selected_features]

# # Train the classifier on the training data with only two features
# clf = svm.NuSVC(gamma='auto')
# clf.fit(train_X_vis, train_y)

# # Create a grid to plot the decision boundary
# h = .02  # step size in the mesh
# x_min, x_max = train_X_vis.iloc[:, 0].min() - 1, train_X_vis.iloc[:, 0].max() + 1
# y_min, y_max = train_X_vis.iloc[:, 1].min() - 1, train_X_vis.iloc[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# # Plot the decision boundary by assigning a color to each point in the mesh
# Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')

# # Plot also the training points
# plt.scatter(train_X_vis.iloc[:, 0], train_X_vis.iloc[:, 1], c=train_y, cmap=plt.cm.coolwarm)
# plt.xlabel(selected_features[0])
# plt.ylabel(selected_features[1])
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.title('SVM Decision Boundary')

# plt.show()
