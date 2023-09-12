from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
from tkinter import filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

#multiple regression Algorithm
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import metrics


def upload():
    global filename
    global df
    filename = filedialog.askopenfilename(initialdir = "dataset")
    pathlabel.config(text=filename)
    df = pd.read_csv(filename)
    text.delete('1.0', END)
    text.insert(END,'dataset loaded\n')
    text.insert(END,"Dataset Size : "+str(len(df))+"\n")

def splitdataset():
    global df, X_train, X_test, y_train, y_test
    X = df.drop(columns="Item_Outlet_Sales", axis=1)
    Y = df["Item_Outlet_Sales"]
    print(X)
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=21)
    text.delete("1.0",END)
    text.insert(END,"Dataset split\n")
    text.insert(END,"Splitted Training Size for Machine Learning : "+str(len(X_train.shape))+"\n")
    text.insert(END,"Splitted Test Size for Machine Learning    : " + str(len(X_test.shape)) + "\n\n")
    text.insert(END, str(X))
    text.insert(END, str(Y))
    return X, Y, X_train, X_test, y_train, y_test


def xgbregressor():
    text.delete('1.0', END)
    xgbr_model = XGBRegressor()
    xgbr_model.fit(X_train, y_train)
    training_data_prediction = xgbr_model.predict(X_train)
    r2_train = metrics.r2_score(y_train, training_data_prediction)
    print("R squared value: ", r2_train*100)
    text.insert(END,"R squared value: "+ str(r2_train*100))


    test_data_prediction = xgbr_model.predict(x_test)
    r2_test = metrics.r2_score(y_test, test_data_prediction)
    print("R squared value:", r2_test)
    text.insert(END,"R squared value: " + str(r2_test*100))

def cal_accuracy(y_test, y_pred, details):
    accuracy = accuracy_score(y_test, y_pred) * 100
    text.insert(END, details + "\n\n")
    return accuracy

def linearregression():
    global classifier
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = LogisticRegression(penalty='l2', dual=False, tol=0.002, C=2.0)
    cls.fit(X_train, y_train)
    text.insert(END, "Prediction Results\n\n")
    prediction_data = prediction(X_test, cls)
    lr_acc = cal_accuracy(y_test, prediction_data, 'Logistic Regression Algorithm Accuracy')
    text.insert(END, "Logistic Regression Accuracy : " + str(lr_acc) + "\n\n")
    classifier = cls


main = tkinter.Tk()
main.title("Predictive Analysis for Big Mart Sales Using Machine Learning Algorithms")
main.geometry("1600x1000")

font = ('times', 16, 'bold')
title = Label(main, text='Predictive Analysis for Big Mart Sales Using Machine Learning Algorithms',font=("times"))
title.config(bg='Dark Blue', fg='white')
title.config(font=font)
title.config(height=3, width=145)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=180)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Crop Recommendation", command=upload)
uploadButton.place(x=50, y=550)
uploadButton.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')
pathlabel.config(font=font1)
pathlabel.place(x=330, y=550)

font1 = ('times', 13, 'bold')
splitdataset = Button(main, text="splitdataset", command=splitdataset)
splitdataset.place(x=100, y=700)
splitdataset.config(font=font1)

font1 = ('times', 13, 'bold')
splitdataset = Button(main, text="xgbregressor", command=xgbregressor)
splitdataset.place(x=300, y=700)
splitdataset.config(font=font1)

font1 = ('times', 13, 'bold')
splitdataset = Button(main, text="Linear Regression", command=linearregression)
splitdataset.place(x=510, y=700)
splitdataset.config(font=font1)

font1 = ('times', 13, 'bold')
splitdataset = Button(main, text="Polynomial regression", command=xgbregressor)
splitdataset.place(x=720, y=700)
splitdataset.config(font=font1)

font1 = ('times', 13, 'bold')
splitdataset = Button(main, text="Ridge regression", command=xgbregressor)
splitdataset.place(x=930, y=700)
splitdataset.config(font=font1)



main.config(bg='turquoise')
main.mainloop()