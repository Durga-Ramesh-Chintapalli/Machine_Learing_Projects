from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter as tk
from tkinter import filedialog

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error


def upload():
    global filename
    global df
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    df = pd.read_csv(filename)
    text.delete('1.0', END)
    text.insert(END, 'dataset loaded\n')
    text.insert(END, "Dataset Size : " + str(len(df)) + "\n")

def splitdataset():
    global df, X_train, X_test, y_train, y_test
    X = df.drop(columns=['charges'])
    Y = df['charges']
    print(X)
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=21)
    text.delete('1.0', END)
    text.insert(END, "Dataset split\n")
    text.insert(END, "Splitted Training Size for Machine Learning : " + str(len(X_train.shape)) + "\n")
    text.insert(END, "Splitted Test Size for Machine Learning    : " + str(len(X_test.shape)) + "\n\n")
    text.insert(END, str(X))
    text.insert(END, str(Y))
    return X, Y, X_train, X_test, y_train, y_test

def Random_forest():
    global Random
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    Random = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=33)
    Random.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n")
    aa=Random.score(X_train, y_train)
    print('Accuracy for the RF is ', str(aa * 100) , '%')
    text.insert(END, "Random Accuracy : " + str(aa * 100) ,"%"+ "\n\n")

def dt():
    global Decision, dt_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    Decision = DecisionTreeRegressor(max_depth=10, random_state=33)
    Decision.fit(X_train, y_train)
    text.insert(END, "Prediction Results\n\n")
    aa=Decision.score(X_train, y_train)
    print('DecisionTreeRegressor Train Score is : ', aa*100)
    text.insert(END,"DT Accuracy: "+str(aa*100)+"%"+ "\n\n")

#Linear regression
def lr():
    global Decision1, dt_acc1
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    Decision1 = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=-1)
    Decision1.fit(X_train, y_train)
    text.insert(END, "Prediction Results\n\n")
    bb=Decision1.score(X_train, y_train)
    print('Linear Regression Train Score is : ', bb*100)
    text.insert(END,"Linear Regression Accuracy: "+str(bb*100)+"%"+ "\n\n")

def predictPerformance():
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="dataset")
    test = pd.read_csv(filename)
    new_test = test.loc[0:]
    survived = 0
    not_Survived = 0
    records = test.values[:, 0:]
    value2 = Random.predict(recods)
    print("result : " + str(value2))
    for i in range(len(value2)):
        if str(value2[i]) == '0':
            text.insert(END, "Not Survived-1-Year\n\n")
            not_Survived += 1  # not_Survived = not_Survived+1
            # print("Not Survived-1-Year")

        else:
            text.insert(END, "Survived-1-Year\n\n")
                # print("Survived-1-Year")

            survived += 1
    print(not_Survived)
    print(survived)


main = tk.Tk()
main.title("Medical Insurence Charges")
main.geometry("1600x1500")

font = ('times', 16, 'bold')
title = Label(main, text='Medical Insurence Charges',font=("times"))
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


splitButton = Button(main, text="Split Dataset", command=splitdataset)
splitButton.place(x=50, y=600)
splitButton.config(font=font1)

#random forest button
ranButton = Button(main, text="Run RF Algorithm", command=Random_forest)
ranButton.place(x=200,y=600)
ranButton.config(font=font1)

#Decision Tree
dtButton = Button(main, text="Decision Tree", command=dt)
dtButton.place(x=400, y=600)
dtButton.config(font=font1)


LRButton = Button(main, text="Linear Regression", command=lr)
LRButton.place(x=530,y=600)
LRButton.config(font=font1)

predictButton = Button(main, text="Predict", command=predictPerformance)
predictButton.place(x=730,y=600)
predictButton.config(font=font1)

main.config(bg='turquoise')
main.mainloop()