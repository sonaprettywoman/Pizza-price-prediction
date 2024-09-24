# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 14:26:34 2024

@author: Dell G3 3579
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df= pd.read_csv(r"C:\Users\Dell G3 3579\Desktop\pizza_v1.csv")
df.head()
df.tail()
df.shape
df.info()
df.isnull().sum()
df.duplicated().any()
df=df.drop_duplicates(keep="first")
df.describe()
df.head()
df.rename({"price_rupiah":"price"},axis=1,inplace= True)

df["price"]=df["price"].str.replace("Rp","")
df["price"]=df["price"].str.replace(",","").astype("int32")
def convert(value):
    return value*0.0054
df["price"].apply(convert)
df["price"]=df["price"].apply(convert)
df["price"]

df.columns
df["company"].value_counts()
sns.displot(x="price",data=df,kde=True)
plt.title("Price Distribution")
plt.show #Right skewed data

df["diameter"].value_counts()#12inches was most ordered
sns.countplot(x=df["diameter"],palette="muted")
plt.show

df["topping"].value_counts()#chic,mush,moz were most fav toppings
sns.countplot(x=df["topping"],palette="gist_ncar")
plt.xticks(rotation=90)
plt.show

df["variant"].value_counts()#classic was most preferred
sns.countplot(x=df["variant"],palette="Accent")
plt.xticks(rotation=90)
plt.show

df["size"].value_counts()
sns.countplot(x=df["size"],palette="BuPu")
plt.show

df["extra_sauce"].value_counts()
sns.countplot(x=df["extra_sauce"],palette="hot")
plt.show

df["extra_cheese"].value_counts()
sns.countplot(x=df["extra_cheese"],palette="autumn")
plt.show

sns.barplot(x="company",y="price",data=df,palette="Dark2")
plt.title("Price vs Company")
plt.show#A has earned more comparitively

sns.boxplot(x="price",y="topping",data=df,palette="Set3")
plt.title("Price vs Topping")
plt.show#Pepporoni cost more than others

sns.barplot(x="size",y="price",data=df,palette="Set1")
plt.title("Price vs Size")
plt.show

df[df["price"].max()==df["price"]]
df[df["size"]=="jumbo"]["diameter"]
df[df["size"]=="XL"]["diameter"]
df[(df["size"]=="jumbo")&(df["diameter"]==16)]
df.drop(df.index[[6,11,16,80]],inplace=True)
df.shape
df.info()
a= df.select_dtypes(include=["object"]).columns
from sklearn.preprocessing import LabelEncoder
e= LabelEncoder()
for i in a:
    df[i]=e.fit_transform(df[i])
df.head()
df.corr()
plt.figure(figsize=(14,6))
sns.heatmap(df.corr(),annot=True,vmin=-1,vmax=1,cmap="GnBu",center=0,robust=True,fmt=".2g",linewidths=2,linecolor="white",cbar=True)

X=df.drop("price",axis=1)
y=df["price"]
X.head()
y.head()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
# pip install xgboost
from xgboost import XGBRegressor

lr=LinearRegression()
lr.fit(X_train,y_train)

svm=SVR()
svm.fit(X_train,y_train)

rfr=RandomForestRegressor()
rfr.fit(X_train,y_train)

gbr=GradientBoostingRegressor()
gbr.fit(X_train,y_train)

xgb=XGBRegressor()
xgb.fit(X_train,y_train)

y_pred1=lr.predict(X_test)
y_pred2=svm.predict(X_test)  
y_pred3=rfr.predict(X_test)
y_pred4=gbr.predict(X_test)
y_pred5=xgb.predict(X_test)

from sklearn import metrics

score1= metrics.r2_score(y_test,y_pred1)
score2= metrics.r2_score(y_test,y_pred2)
score3= metrics.r2_score(y_test,y_pred3)
score4= metrics.r2_score(y_test,y_pred4)
score5= metrics.r2_score(y_test,y_pred5)

print(score1,score2,score3,score4,score5)


sum_model=pd.DataFrame({"Model":["LR","SVR","RFR","GBR","XGR"],"R2_Score":[score1,score2,score3,score4,score5]})
sum_model

plt.figure(figsize=(10,10))
sns.barplot(x=sum_model["Model"],y=sum_model["R2_Score"],data=sum_model,palette="gist_rainbow",width=0.5 )
plt.title("Best Score")
plt.show()#XGB model gives best score

rfr.feature_importances_
fi=pd.Series(rfr.feature_importances_,index=X_train.columns)
fi
fi.plot(kind="barh")

gbr.feature_importances_
fi=pd.Series(gbr.feature_importances_,index=X_train.columns)
fi.plot(kind="barh")

xgb.feature_importances_
fi=pd.Series(xgb.feature_importances_,index=X_train.columns)
fi.plot(kind="barh")

X=df.drop("price",axis=1)
y=df["price"]

xgb=XGBRegressor()
xgb.fit(X,y)

import joblib
joblib.dump(xgb,"pizza_price_prediction")
model= joblib.load("pizza_price_prediction")

df=pd.DataFrame({"company":1,
                 "diameter":22.0,
                 "topping":2,
                 "variant":8,
                 "size":1,
                 "extra_sauce":1,
                 "extra_cheese":1},index=[0])
df
model.predict(df)


from tkinter import *
import joblib

master= Tk()
master.title("Pizza Price Prediction Using Machine Learning")
label = Label(master,text = "Pizza Price Prediction",bg = "black",
               fg = "white").grid(row=0,columnspan=2)

Label(master,text = "Company Name").grid(row=1)
Label(master,text = "Enter the diameter").grid(row=2)
Label(master,text = "Topping").grid(row=3)
Label(master,text = "Variant").grid(row=4)
Label(master,text = "Size").grid(row=5)
Label(master,text = "extra_sauce[1:yes,0:no]").grid(row=6)
Label(master,text = "extra_cheese[1:yes,0:no]").grid(row=7)

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
 
 
   
   
e1.grid(row=1,column=1)
e2.grid(row=2,column=1)
e3.grid(row=3,column=1)
e4.grid(row=4,column=1)
e5.grid(row=5,column=1)
e6.grid(row=6,column=1)
e7.grid(row=7,column=1)


result_var = StringVar()
result_var.set("Pizza price will be shown here")

Label(master, text="Pizza price is").grid(row=31)
result_label = Label(master, textvariable=result_var)
result_label.grid(row=32)   
    

# df=pd.DataFrame({"company":p1,
#                     "diameter":p2,
#                     "topping":p3,
#                     "variant":p4,
#                     "size":p5,
#                     "extra_sauce":p6,
#                     "extra_cheese":p7},index=[0])
# df
model= joblib.load("pizza_price_prediction")

def predict_price():
    
    try:
        p1=float(e1.get())
        p2=float(e2.get())
        p3=float(e3.get())
        p4=float(e4.get())
        p5=float(e5.get())
        p6=float(e6.get())
        p7=float(e7.get())  
        result=model.predict([[p1,p2,p3,p4,p5,p6,p7]])
        result_var.set(f"₹{result[0]:.2f}")
    except ValueError:
        result_var.set("Please enter valid numbers")
       
    
#print(predict_price())


#Label(master,text="Pizza price is").grid(row=31)





Button(master,text="Predict",command=predict_price).grid(row=33)
#Label(master,text=result).grid(row=32)
mainloop()




