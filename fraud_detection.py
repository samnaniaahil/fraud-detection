# Fraud Detection in Banks with SOMs

# Part 1 - Identify frauds with a Self-Organizing Map (SOM)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv(r'C:\Users\firoj\Downloads\P16-Self-Organizing-Maps\Self_Organizing_Maps\Credit_Card_Applications.csv')
# Seperate dataset into dependent and independent variable(s)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
# Normalize independent variables to values betweeen 0 and 1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
# Initialize parameters 
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
# Randomly initialize weights
som.random_weights_init(X)
# Train SOM on independent variables
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
# Intialize window for SOM
bone()
# Rerturn transposed (flipped over diagonal) matrix for all mean inter-neuron distances (MID)
pcolor(som.distance_map().T)
# Create legend to determine what darkness of colour corresponds with a low MID or a high MID
colorbar()
# Create markers to determine if a customer got apporved (green squares) or not (red circles)
markers = ['o', 's']
colors = ['r', 'g']
# Get winning node for each customer
for i, x in enumerate(X):
    w = som.winner(x)
    # Plot a marker on winning node depending on if the customer gets approved
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]], markeredgecolor = colors[y[i]], markerfacecolor = 'None', markersize = 10, markeredgewidth = 2)
show()

# Get a list of customers with fradulent activity
mappings = som.win_map(X)
# Get coordinates of outlying winning nodes
frauds = np.concatenate((mappings[(5,3)], mappings[(8,3)]), axis = 0)
# Inverse normailization
frauds = sc.inverse_transform(frauds)


# Part 2 - Creating matrix of features (independent varibles) and creating dependent variable

# Creating matrix of features 
customers = dataset.iloc[:, 1:].values

# Creating the dependent variable
# Initialize vector of zeros
is_fraud = np.zeros(len(dataset))
# If a customer ID is found in list of frauds, change its value from 0 to 1
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)


# Part 3 - Making an ANN

# Importing Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer with Dropout
classifier.add(Dense(output_dim = 2, init = 'uniform', activation = 'relu', input_dim = 15))
# Droput regularization for potential overfitting
classifier.add(Dropout(0.1))

# Droput regularization for potential overfitting
classifier.add(Dropout(0.1))

# Adding output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 

# Fitting the ANN to the training set
classifier.fit(customers, is_fraud, batch_size = 1, nb_epoch = 2)

# Predict each probabilty of each customer bieng a fraud
y_pred = classifier.predict(customers)

# Create 2-d array of of customer IDs and predicted probabilites
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)

# Sort customers from least to most likely to be a fraud
y_pred = y_pred[y_pred[:, 1].argsort()]