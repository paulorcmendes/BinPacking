# -*- coding: utf-8 -*-
"""
Created on Fri Jun 08 23:40:34 2018

@author: paulo
"""
import numpy as np

A = [4,3,2,3]
lenA = len(A)
V = 5

def fitness(chrm):
       
    fit = 0
    aux = np.reshape(chrm, (lenA, lenA))
    M1 = np.full((lenA, 1),1)    
    print aux    
    if np.matmul(aux,M1) == M1 and np.matmul(A,aux) <= V:
        fit+=lenA*10 #só colocando pra ja ficar bem melhor que as que violam
        for i in range lenA:
            for j in range lenA:
                if aux[j][i] == 1:
                    break
            if j == lenA:
                fit+=1
    return fit
        




chrm = np.random.randint(0, 2, lenA**2)




    
    
        
