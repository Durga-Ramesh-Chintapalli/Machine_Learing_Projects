from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

main = tkinter.Tk()
main.title("Stroke Risk Prediction With Hybrid Deep Transfer Learning Framework")
main.geometry("1600x1500")

global filename
global svm_mae, random_mae, logistic_mae
global X, Y, X_train, X_test, y_train, y_test
global classifier
global df


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

    X = df.drop(columns=['stroke'])
    Y = df['stroke']
    print(X)
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=21)
    return X, Y, X_train, X_test, y_train, y_test


def matrixs():
    global X, Y, X_train, X_test, y_train, y_test
    X, Y, X_train, X_test, y_train, y_test = splitdataset()
    text.delete('1.0', END)
    text.insert(END, "Splitted Training Size for Machine Learning : " + str(len(X_train.shape)) + "\n")
    text.insert(END, "Splitted Test Size for Machine Learning    : " + str(len(X_test.shape)) + "\n\n")
    text.insert(END, str(X))


def prediction(X_test, cls):
    y_pred = cls.predict(X_test)
    for i in range(len(X_test)):
        print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred


def cal_accuracy(y_test, y_pred, details):
    accuracy = accuracy_score(y_test, y_pred) * 100
    text.insert(END, details + "\n\n")
    return accuracy


def sc():
    global X, Y, X_train, X_test, y_train, y_test
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    text.insert(END, "sc done")


def logisticRegression():
    global classifier
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = LogisticRegression(penalty='l2', dual=False, tol=0.002, C=2.0)
    cls.fit(X_train, y_train)
    text.insert(END, "Prediction Results\n\n")
    prediction_data = prediction(X_test, cls)
    lr_acc = cal_accuracy(y_test, prediction_data, 'Logistic Regression Algorithm Accuracy')
    text.insert(END, "Logistic Regression Algorithm Accuracy : " + str(lr_acc) + "\n\n")
    classifier = cls


def SVM():
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = svm.SVC(kernel='linear')
    cls.fit(X_train, y_train)
    text.insert(END, "Prediction Results\n\n")
    prediction_data = prediction(X_test, cls)
    svm_acc = cal_accuracy(y_test, prediction_data, 'SVM Algorithm Accuracy')
    text.insert(END, "SVM Accuracy : " + str(svm_acc) + "\n\n")


def dt():
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = DecisionTreeClassifier(criterion='entropy', random_state=0)
    cls.fit(X_train, y_train)
    text.insert(END, "Prediction Results\n\n")
    prediction_data = prediction(X_test, cls)
    dt_acc = cal_accuracy(y_test, prediction_data, 'DT Algorithm Accuracy')
    text.insert(END, "DT Accuracy : " + str(dt_acc) + "\n\n")


def predictPerformance():
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="dataset")
    test = pd.read_csv(filename)
    records = test.values[:, 0:22]
    value = classifier.predict(records)
    print("result : " + str(value))
    for i in range(len(value)):
        if str(value[i]) == '0':
            print("Not Survived-1-Year")
        else:
            print("Survived-1-Year")


font1 = ('times', 13, 'bold')
text = Text(main, height=25, width=175)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set, relief="solid", borderwidth=2, highlightbackground="blue")
text.place(x=10, y=100)
text.config(font=font1)

font = ('times', 18, 'bold')
title = Label(main, text='Predicting Employees under Stress for Pre-emptive Remediation using Machine learning Algorithm')
title.config(bg='firebrick4', fg='dodger blue')
title.config(font=font)
title.config(height=3, width=140)
title.place(x=0, y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Dataset", command=upload)
upload.place(x=100, y=620)
upload.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')
pathlabel.config(font=font1)
pathlabel.place(x=250, y=625)

font1 = ('times', 13, 'bold')
splitdataset = Button(main, text="splitdataset", command=splitdataset)
splitdataset.place(x=100, y=700)
splitdataset.config(font=font1)

font1 = ('times', 13, 'bold')
matrixs = Button(main, text="matrixs", command=matrixs)
matrixs.place(x=210, y=700)
matrixs.config(font=font1)

scc = Button(main, text="StandardScaler", command=sc)
scc.place(x=400, y=700)
scc.config(font=font1)

log = Button(main, text="Run Logistic Regression Algorithm", command=logisticRegression)
log.place(x=550, y=700)
log.config(font=font1)

svmButton = Button(main, text="Run SVM Algorithm", command=SVM)
svmButton.place(x=850, y=700)
svmButton.config(font=font1)


dtButton = Button(main, text="Run DT Algorithm", command=dt)
dtButton.place(x=1050, y=700)
dtButton.config(font=font1)

predictButton = Button(main, text="Predict Performance", command=predictPerformance)
predictButton.place(x=1250, y=700)
predictButton.config(font=font1)

main.config(bg='turquoise')
main.mainloop()