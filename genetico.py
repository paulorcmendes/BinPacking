# -*- coding: utf-8 -*-
"""
Created on Fri Jun 08 23:40:34 2018

@author: paulo
"""
import numpy as np

def BinToInt(nBin):
    return int(''.join(str(x) for x in nBin), 2)

def fitness(chrm, A, sizePerEntry):       
    fit = 1    
    MChrm = np.reshape(chrm, (len(A), sizePerEntry))
    
    X = np.full((len(A),len(A)),0)
    for i in range(len(A)):
        X[i][int(np.around((len(A)-1)*float(BinToInt(MChrm[i]))/((2**sizePerEntry) - 1)))] = 1
       
    M1 = np.full((len(A), 1),1)   
     
    if (np.matmul(X,M1) == M1).all() and (np.matmul(A,X) <= V).all():
        #A primeira restrição nunca é violada, btw
        fit+=len(A)*3 #só colocando pra ja ficar bem melhor que as que violam
        for i in range(len(A)):
            has1 = False
            for j in range(len(A)):
                if X[j][i] == 1:
                    has1 = True
                    break
            if has1 == False:
                fit+=2
    return fit
        
A = [4,3,2,4]
lenA = len(A)
V = 7


sizePerEntry = int(np.log2(lenA-1))*5
chrm = np.random.randint(0, 2, lenA*sizePerEntry)

print fitness(chrm, A, sizePerEntry)    
