# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 21:06:03 2024

@author: Dell G3 3579
"""

import random

chances=0
a= random.randint(1, 100)


for x in range(a):
    x= int(input("Enter your number: "))
    chances+=1
    if x<1 or x>100:
        print("Only choose numbers between 1 to 100")
    elif x<a:
        print("Very less,take one more guess")
    elif x>a:
         print("You need to go lower,the game ain't over")
    else:
        print("Congratulations!!,you got it right!!")
        break
print("End of the game")        