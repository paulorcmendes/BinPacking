# -*- coding: utf-8 -*-
"""
Created on Fri Jun 08 23:40:34 2018

@author: paulo
"""
import numpy as np
import matplotlib.pyplot as plt

#creates a new population
def NewPop(POP_SIZE, CHRM_SIZE):
    return np.random.randint(0, 2, size=(POP_SIZE, CHRM_SIZE))
#function that returns the finesses of an entire population in the form of an array
def PopFitness(pop, SIZEPERENTRY):
    return [Fitness(chrm, SIZEPERENTRY) for chrm in pop]
#roullete
def RoulleteSelection(fitness):
    perc = np.array(fitness)/sum(fitness)
    total = 0.0
    for i in range(len(perc)):
        perc[i] += total
        total = perc[i]
    s = np.random.random()
    for i in range(len(perc)):
        if s <= perc[i]:
            return i
#crossing two chromossomes        
def Crossover(father, mother, CHRM_SIZE):
    limit = np.random.randint(CHRM_SIZE)
    return (list(father[:limit])+list(mother[limit:]),list(mother[:limit])+list(father[limit:]))
#changing a bit in a random position
def Mutation(chrm, CHRM_SIZE):
    pos = np.random.randint(CHRM_SIZE)
    chrm[pos] = 1 - chrm[pos]
    return chrm


def BinToInt(nBin):
    return int(''.join(str(x) for x in nBin), 2)

def Fitness(chrm, SIZEPERENTRY):       
    fit = len(A)
    MChrm = np.reshape(chrm, (len(A), SIZEPERENTRY))
    
    X = np.full((len(A),len(A)),0)
    for i in range(len(A)):
        X[i][int(np.around((len(A)-1)*float(BinToInt(MChrm[i]))/((2**SIZEPERENTRY) - 1)))] = 1
    #print X   
    M1 = np.full((len(A), 1),1)   
     
    if (np.matmul(X,M1) == M1).all() and (np.matmul(A,X) <= V).all():
        #A primeira restrição nunca é violada, btw
        fit+=len(A)*4 #só colocando pra ja ficar bem melhor que as que violam
        for i in range(len(A)):
            has1 = False
            for j in range(len(A)):
                if X[j][i] == 1:
                    has1 = True
                    break
            if has1 == False:
                fit+=2
    return float(fit)
        
def Obj(fit):
    bestObj = len(A)-(fit-5*len(A))/2
    if bestObj > len(A):
        return len(A)+1
    else:
        return bestObj
def PopObj(fitness):
    return [Obj(fit) for fit in fitness]

#chrm = np.random.randint(0, 2, CHRM_SIZE)

#print obj(fitness(chrm))    
def GeneticAlgorithm(A, V, name):
    SIZEPERENTRY = int(np.log2(len(A)-1))*5
    CROSSOVER_RATE = 0.9 #RATE OF CROSSOVER
    MUTATION_RATE = 0.001 #RATE OF MUTATION
    POP_SIZE = 100 #POPULATION SIZE
    N_GENERATIONS = 100 #MAXIMUM NUMBER OF GENERATIONS
    CHRM_SIZE = len(A)*SIZEPERENTRY
    pop = NewPop(POP_SIZE, CHRM_SIZE)
    best = []
    obj = []
    mean = []
    for gen in range(N_GENERATIONS):   
        fitness = PopFitness(pop, SIZEPERENTRY)
        popObj = PopObj(fitness)
        mean.append(np.mean(popObj))
        
        best.append(np.max(fitness))
        obj.append(Obj(best[-1]))
        
        #<elitism>
        bestChrm = pop[fitness.index(best[-1])]
        #</elitism>
        nextPop = []    
        for i in range(POP_SIZE/2):
            father = pop[RoulleteSelection(fitness)]
            mother = pop[RoulleteSelection(fitness)]
            
            tx = np.random.random()
            if tx <= CROSSOVER_RATE:           
                son, daughter = Crossover(father, mother, CHRM_SIZE)
            else:
                son, daughter = father, mother
            tx = np.random.random()
            if tx <= MUTATION_RATE:
                son = Mutation(son, CHRM_SIZE)
            tx = np.random.random()
            if tx <= MUTATION_RATE:
                daughter = Mutation(daughter, CHRM_SIZE)
            nextPop.append(son)
            nextPop.append(daughter)
        pop = nextPop
        #elitism
        auxF = PopFitness(pop, SIZEPERENTRY)
        pop[auxF.index(np.min(auxF))] = bestChrm
        #</elitism>
    fitness = PopFitness(pop, SIZEPERENTRY)
    popObj = PopObj(fitness)
    mean.append(np.mean(popObj))
    best.append(np.max(fitness))
    obj.append(Obj(best[-1]))
    
    
    x = np.arange(0.0, N_GENERATIONS+1, 1.0)
    
    print "Problem: "+name
    plt.plot(x, obj)
    plt.plot(x, mean)
    plt.show()
    print "Solution: "+str(obj[-1])

A = [4,3,2,5, 1, 3, 4, 5, 2, 1, 4, 3]
V = 5
GeneticAlgorithm(A, V, "teste")
