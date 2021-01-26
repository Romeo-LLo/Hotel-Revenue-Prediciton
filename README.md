# NTU Machine Learning Final Project
With the training data provided, this work illustrated the SVM models for predicting the daily revenue of specific hotel booking platform.


## Data Preprocessing(Data_preprocess.py)
In consideration of effectiveness, MinMaxScaler is applied to the numerical data, while target encoding is adopted for the catogorical data.

## Prediction(prediction.py)

The prediction is seperated into two parts, 
1. Predict whether the deal is canceled using SVC
2. Predict how much the deal earns with SVR

The model could easily be replaced with any model since the x_train and y_train is provided in this function already.

## To the final prediction Label(main.py)
Revenue lies between 1 and 10000 will be labeled 1, 10001 and 20000 becomes 2, and so on. The prediciton is grouped by date and then summed up. 

Apart from generating y_test, validation result are also avaliable, which helps to evaluate the model.