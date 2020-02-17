import warnings
warnings.filterwarnings("ignore")
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
import numpy as np

##Import datasets and modify them##
train_data = pd.read_csv('intern_data.csv')
test_data = pd.read_csv('intern_test.csv')
del(train_data['Unnamed: 0'])
del(test_data['Unnamed: 0'])

#Replace 'string' data with floats#
mapping_c = {'green':0,'red':1,'yellow':2,'blue':3}
mapping_h = {'white':0,'black':1}
train_data = train_data.replace({'c':mapping_c,'h':mapping_h})
train_data['c'] = train_data['c'].astype(float)
train_data['h'] = train_data['h'].astype(float)

test_data = test_data.replace({'c':mapping_c,'h':mapping_h})
test_data['c'] = test_data['c'].astype(float)
test_data['h'] = test_data['h'].astype(float)

del(mapping_c,mapping_h)

##Use the TSNE model to reduce the dimensions of teh data and visualize how##
##the variables are distributed##
model = TSNE()
model_transformed = model.fit_transform(train_data)
xs = model_transformed[:,0]
ys = model_transformed[:,1]
plt.scatter(xs,ys)
plt.show()

#Transform the data points to see if we can find a linear relationship# 
#amongst the clusters#
plt.scatter(np.square(xs),np.square(ys))
plt.show()

##Determine if there are correlations between variables in the data##
eda = pd.plotting.scatter_matrix(train_data)
print(eda)
del(eda)

##Partition the training set into target and feature variables##
train_y = pd.DataFrame(train_data['y'])
del(train_data['y'])

#Split the training set to test the linear regression model later#
x_train,x_test,y_train,y_test = train_test_split(train_data,train_y,
                                                 test_size = 0.1,random_state = 12)

##Run a regression model using a Support Vector Machine on the training set##
##and check its accuracy##
svr = SVR(kernel = 'rbf',gamma = 0.5)
svr_predict = svr.fit(x_train,y_train).predict(x_test)
print("\nSVR Training Score: ", svr.score(x_train,y_train))
print("\nSVR Test Score: ", svr.score(x_test,y_test))

#Cross-Validation Score of the SVR#
cvs = cross_val_score(svr, train_data, train_y, cv = 5)
print("\nCross Validation Scores: ", cvs)

#Plot the tru target values against the predicted ones to visualize#
#how well the regression model did#
fig, ax = plt.subplots()
ax.scatter(y_test, svr_predict, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

del(x_train,y_train,x_test,y_test)
del(ax,fig)

##Use the regression model to predict the target values for the test set##
svr.fit(train_data,train_y)
prediction = svr.predict(test_data).tolist()

#This bit is just so I can combine the indices of the test set with the#
#predictions#  
test_data = pd.read_csv('intern_test.csv')
prediction_final = pd.DataFrame(test_data['Unnamed: 0'])
prediction_final.columns = ['i']
prediction_final['y'] = prediction
prediction_final.set_index('i', inplace = True)

#Export to a .csv file#
prediction_final.to_csv('intern_predicted.csv')













































