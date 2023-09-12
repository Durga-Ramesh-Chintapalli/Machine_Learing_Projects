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
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

main = tkinter.Tk()
main.title("Patient Survival After One Year of Treatment")
main.geometry("1300x1200")

global filename
global X, Y, X_train, X_test, y_train, y_test
global model
global df
global lr_acc,svm_acc,dt_acc,random_acc
global survived,not_Survived


def upload():
    global filename
    global df
    filename = filedialog.askopenfilename(initialdir = "dataset")
    pathlabel.config(text=filename)
    df = pd.read_csv(filename)
    text.delete('1.0', END)
    text.insert(END,'dataset loaded\n')
    text.insert(END,"Dataset Size : "+str(len(df))+"\n")


def matrix():
    global X, Y, X_train, X_test, y_train, y_test
    X = df.drop(columns=['Survived_1_year'])
    Y = df['Survived_1_year']
    #X = matrix_factor.values[:, 1:-1] 
    #Y = matrix_factor.values[:, -1]
    print("=======================>",X)
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=21)
    
    text.delete('1.0', END)
    text.insert(END,"Splitted Training Size for Machine Learning : "+str(len(X_train.shape))+"\n")
    text.insert(END,"Splitted Test Size for Machine Learning    : "+str(len(X_test.shape))+"\n\n")
    text.insert(END,str(X))

def prediction(X_test, cls): 
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, details): 
    accuracy = accuracy_score(y_test,y_pred)*100
    text.insert(END,details+"\n\n")
    return accuracy  
    
def sc():
      global X, Y, X_train, X_test, y_train, y_test
      sc = StandardScaler()
      X_train = sc.fit_transform(X_train)
      X_test = sc.transform(X_test)
      text.insert(END,"sc done")

def logisticRegression():
    global logistic,lr_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    logisitc = LogisticRegression(penalty='l2', dual=False, tol=0.002, C=2.0)
    logisitc.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, logisitc) 
    lr_acc = cal_accuracy(y_test, prediction_data,'Logistic Regression Algorithm Accuracy')
    text.insert(END,"Logistic Regression Algorithm Accuracy : "+str(lr_acc)+"\n\n")


def SVM():
    global support,svm_acc 
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    support = svm.SVC(kernel = 'linear') 
    support.fit(X_train, y_train) 
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, support) 
    svm_acc = cal_accuracy(y_test, prediction_data,'SVM Algorithm Accuracy')
    text.insert(END,"SVM Accuracy : "+str(svm_acc)+"\n\n")


def dt():
    global Decision,dt_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    Decision= DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    Decision.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, Decision) 
    dt_acc = cal_accuracy(y_test, prediction_data,'DT Algorithm Accuracy')
    text.insert(END,"DT Accuracy : "+str(dt_acc)+"\n\n")
def ran():
    global Random,random_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    Random= RandomForestClassifier(n_estimators= 10, criterion="entropy")  
    Random.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, Random) 
    random_acc = cal_accuracy(y_test, prediction_data,'Random Algorithm Accuracy')
    text.insert(END,"Random Accuracy : "+str(random_acc)+"\n\n")

def predictPerformance():
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "dataset")
    test = pd.read_csv(filename)
    new_test= test.loc[0:2999]
    survived=0
    not_Survived =0
    records= test.values[:,0:21]
    value2 = Random.predict(records)
    print("result : "+str(value2))
    for i in range(len(value2)):
        if str(value2[i])=='0':
            text.insert(END,"Not Survived-1-Year\n\n")
            not_Survived+=1 # not_Survived = not_Survived+1
            #print("Not Survived-1-Year")

        else:
            text.insert(END,"Survived-1-Year\n\n")
            #print("Survived-1-Year") 
            survived+=1        
    print(not_Survived)
    print(survived)

    #graph
    count=[]
    count.append(survived)
    count.append(not_Survived)
    print(count)
    method = ["Survived","Not Survived"]
    colors = ["green","orange"]
    sns.set_style("whitegrid")
    plt.figure(figsize=(5,5))
    plt.yticks(np.arange(0,10000,1000))
    plt.ylabel("Count")
    plt.xlabel("Label")
    sns.barplot(x=method, y=count, palette=colors)
    plt.show()

def graph():
    print(lr_acc)
    ac=[]
    ac.append(lr_acc)
    ac.append(svm_acc)
    ac.append(dt_acc)
    ac.append(random_acc)
    methods = ["Logistic Regression","Support Vector Mechine","Decision Tree","Random"]
    colors = ["purple", "green","orange","blue"]
    sns.set_style("whitegrid")
    plt.figure(figsize=(5,5))
    plt.yticks(np.arange(0,100,10))
    plt.ylabel("Accuracy %")
    plt.xlabel("Algorithms")
    sns.barplot(x=methods, y=ac, palette=colors)
    plt.show()

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

font = ('times', 16, 'bold')
title = Label(main, text='Patient Survival After One Year of Treatment')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Dataset",command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

matrixButton = Button(main, text="Matrix Factorization", command=matrix)
matrixButton.place(x=700,y=200)
matrixButton.config(font=font1) 

scButton = Button(main, text="StandardScaler", command=sc)
scButton.place(x=700,y=250)
scButton.config(font=font1) 

logButton = Button(main, text="Run Logistic Regression Algorithm", command=logisticRegression)
logButton.place(x=700,y=300)
logButton.config(font=font1)

svmButton = Button(main, text="Run SVM Algorithm", command=SVM)
svmButton.place(x=700,y=350)
svmButton.config(font=font1)

dtButton = Button(main, text="Run DT Algorithm", command=dt)
dtButton.place(x=700,y=400)
dtButton.config(font=font1)

ranButton = Button(main, text="Run Ran Algorithm", command=ran)
ranButton.place(x=700,y=450)
ranButton.config(font=font1)

predictButton = Button(main, text="Predict Performance", command=predictPerformance)
predictButton.place(x=700,y=500)
predictButton.config(font=font1)

"""outputButton = Button(main, text="Count Of Survied or Not", command=outputgraph)
outputButton.place(x=700,y=550)
outputButton.config(font=font1)"""

graphButton = Button(main, text="Model performance Graph", command=graph)
graphButton.place(x=700,y=550)
graphButton.config(font=font1)



main.config(bg='turquoise')
main.mainloop()
