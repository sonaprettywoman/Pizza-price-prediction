# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 21:39:16 2024

@author: Dell G3 3579
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
df=pd.read_excel(r"C:\Users\Dell G3 3579\Desktop\Coffee+Shop+Sales\Coffee Shop Sales.xlsx")
df
df.info()



plt.hist(df['unit_price'],bins=[0,10,20,30,40],log=True)
plt.title("Unit Price")


plt.figure(figsize=(18,12))
sns.barplot(data=df,x='store_location',y='transaction_qty',hue='product_category',)
plt.xlabel("store location",fontsize=30)
plt.xticks(fontsize=30)
plt.ylabel("transaction_qty",fontsize=30)
plt.yticks(fontsize=30)
plt.legend(bbox_to_anchor=(1.05,1.0),loc='upper left',fontsize=25)

