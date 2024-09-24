# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 23:23:10 2024

@author: Dell G3 3579
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
df=pd.read_excel(r"C:\Users\Dell G3 3579\Desktop\Coffee+Shop+Sales\Coffee Shop Sales.xlsx")
df
df.info()

plt.pie(df['transaction_qty'],autopct="%1.2f%%")#logically incorrect


df.groupby("product_type").sum("transaction_qty")
df
df.info()
newdf = df.iloc[:,3:]
newdf


anss = newdf.groupby("product_type").sum()
plt.pie(anss['transaction_qty'],autopct="%1.2f%%")

anss


plt.figure(figsize=(8,8),dpi=150)
sns.scatterplot(data=df,x='unit_price',y='transaction_qty',hue='product_type',markers='+',s=500)
plt.xticks(fontsize=30)
plt.xlabel("unit price",fontsize=32)
plt.ylabel("transaction_qty",fontsize=32)
plt.yticks(fontsize=30)
plt.legend(bbox_to_anchor=(1.05,1.0),loc='upper left',fontsize=15)


sns.lineplot(data=df,x='transaction_date',y='transaction_qty',hue='store_location',ci=None)
plt.rc({'Font'})
plt.legend(bbox_to_anchor=(1.05,1.0),loc='lower left')

plt.figure(figsize=(16,12),dpi=150)
# coffee=df.head(65000)
sns.lineplot(data=df,x='transaction_date',y='transaction_qty',hue='store_location',ci=None)
plt.xticks(fontsize=30)
plt.xlabel('transaction_date',fontsize=30)
plt.ylabel("transaction_qty",fontsize=32)
plt.yticks(fontsize=30)
plt.legend(bbox_to_anchor=(1.05,1.0),loc='lower left',fontsize=15)
