# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:27:48 2024

@author: Dell G3 3579
"""




from bs4 import BeautifulSoup as bs
import requests as re
import csv
url=input("Enter your url")
ps= re.get(url)

soup=bs(ps.text,"html.parser")
quotes=soup.find_all("span",attrs={"class":"text"})
authors=soup.find_all("small",attrs={"class":"author"})

file=open("scrapped quotes.csv","w")
writer=csv.writer(file)

writer.writerow(["Quotes","Authors"])

for quote,author in zip(quotes,authors):
    print(quote.text+"-"+author.text)
    writer.writerow([quote.text,author.text])
file.close()    
    



