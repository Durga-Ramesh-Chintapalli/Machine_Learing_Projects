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
from sklearn.ensemble import RandomForestClassifier  # For classification tasks
from sklearn.ensemble import RandomForestRegressor

main = tkinter.Tk()
main.title("Campus Placements Prediction & Analysis using Machine Learning")
main.geometry("1600x1500")

global filename
global svm_mae,random_mae,logistic_mae
global X, Y, X_train, X_test, y_train, y_test
global classifier
global df
global placed,not_placed
global lr_acc

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
    X = df.drop(columns=['status'])
    Y = df['status']
    print(X)
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=21)
    text.delete('1.0', END)
    text.insert(END, "Dataset split\n")
    text.insert(END, "Splitted Training Size for Machine Learning : " + str(len(X_train.shape)) + "\n")
    text.insert(END, "Splitted Test Size for Machine Learning    : " + str(len(X_test.shape)) + "\n\n")
    text.insert(END, str(X))

    text.insert(END, "Splitted Training Size for Machine Learning : " + str(len(y_train.shape)) + "\n")
    text.insert(END, "Splitted Test Size for Machine Learning    : " + str(len(y_test.shape)) + "\n\n")
    text.insert(END, str(Y))

    return X, Y, X_train, X_test, y_train, y_test


# def matrixs():
#     global X, Y, X_train, X_test, y_train, y_test
#     X, Y, X_train, X_test, y_train, y_test = splitdataset(df)
#     text.delete('1.0', END)
#     text.insert(END,"Splitted Training Size for Machine Learning : "+str(len(X_train.shape))+"\n")
#     text.insert(END,"Splitted Test Size for Machine Learning    : "+str(len(X_test.shape))+"\n\n")
#     text.insert(END,str(X))



def prediction(X_test, cls): 
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred


from sklearn.preprocessing import StandardScaler
def sc():
    global X_train, X_test
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    text.insert(END, "\n\n")
    text.insert(END, "Standardization (sc) done.")

def logisticRegression():
    global classifier, lr_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = LogisticRegression(penalty='l2', dual=False, tol=0.002, C=2.0)
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    lr_acc = accuracy_score(y_test, prediction_data,'Logistic Regression Algorithm Accuracy')
    text.insert(END,"Logistic Regression Algorithm Accuracy : "+str(lr_acc*100)+"\n\n")
    classifier = cls


def RF():
    global Random, random_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)

    Random = RandomForestClassifier(n_estimators=10, criterion="entropy")
    Random.fit(X_train, y_train)
    text.insert(END, "Prediction Results\n\n")
    y_pred2 = Random.predict(X_test)
    random_acc = accuracy_score(y_test, y_pred2)

    print('Accuracy for the RF is ', random_acc * 100, '%')
    text.insert(END, "Random Accuracy : " + str(random_acc * 100) + "\n\n")

def dt():
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls= DecisionTreeClassifier(criterion = 'entropy', random_state = 45)
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    dt_acc = accuracy_score(y_test, prediction_data,'DT Algorithm Accuracy')
    text.insert(END,"DT Accuracy : "+str(dt_acc*100)+"\n\n")


def predictPerformance():
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "dataset")
    test = pd.read_csv(filename)
    new_test= test.loc[0:]
    placed=0
    not_placed =0
    records= test.values[:,0:]
    value2 = Random.predict(records)
    print("result : "+str(value2))
    for i in range(len(value2)):
        if str(value2[i])=='0':
            text.insert(END,"Not placed\n\n")
            not_placed += 1

        else:
            text.insert(END,"placed \n\n")
            # print(" placed")
            placed+=1
    print(not_placed)
    print(placed)

    #graph
    count=[]
    count.append(placed)
    count.append(not_placed)
    print(count)
    method = ["placed","Not placed"]
    colors = ["green","orange"]
    sns.set_style("whitegrid")
    plt.figure(figsize=(5,5))
    plt.yticks(np.arange(0,51,51))
    plt.ylabel("Count")
    plt.xlabel("Label")
    sns.barplot(x=method, y=count, palette=colors)
    plt.show()

def graph():
    print(lr_acc)
    ac=[]
    ac.append(lr_acc)
    ac.append(random_acc)
    ac.append(dt_acc)
    methods=["Logistic Regression","Random Forest","Decision Tree"]
    colors = ["purple", "green", "orange"]
    sns.set_style("whitegrid")
    plt.figure(figsize=(5, 5))
    plt.yticks(np.arange(0, 100, 13))
    plt.ylabel("Accuracy %")
    plt.xlabel("Algorithms")
    sns.barplot(x=methods,y=ac,palette=colors)
    plt.show()

font1 = ('times', 13, 'bold')
text=Text(main,height=25,width=170)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set, relief="solid", borderwidth=2, highlightbackground="blue")
text.place(x=10,y=100)
text.config(font=font1)

font = ('times', 18, 'bold')
title = Label(main, text='Campus Placements Prediction & Analysis using  Machine Learning',font=("times",20))
title.config(bg='silver', fg='black')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Dataset",command=upload)
upload.place(x=100,y=620)
upload.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')
pathlabel.config(font=font1) 
pathlabel.place(x=250,y=625)

font1 = ('times', 13, 'bold')
splitdataset = Button(main, text="splitdataset", command=splitdataset)
splitdataset.place(x=100,y=700)
splitdataset.config(font=font1)


scc = Button(main, text="StandardScaler", command=sc)
scc.place(x=250,y=700)
scc.config(font=font1) 


log = Button(main, text="Run Logistic Regression Algorithm", command=logisticRegression)
log.place(x=400,y=700)
log.config(font=font1)

RFButton = Button(main, text="Run RF Algorithm", command=RF)
RFButton.place(x=720,y=700)
RFButton.config(font=font1)

dtButton = Button(main, text="Run DT Algorithm", command=dt)
dtButton.place(x=900,y=700)
dtButton.config(font=font1)

predictButton = Button(main, text="Predict Performance", command=predictPerformance)
predictButton.place(x=1100,y=700)
predictButton.config(font=font1)

graphButton = Button(main, text="Model performance Graph", command=graph)
graphButton.place(x=1350,y=700)
graphButton.config(font=font1)


main.config(bg='turquoise')
main.mainloop()