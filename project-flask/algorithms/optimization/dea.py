import math
import numpy as np
import matplotlib.pyplot as pyplot
import random as rand
from random import sample


def minimize(cost_func, bounds, dimension, popsize, mutate, recombination, maxiter, initialCF):
    average_cost_value = [0] * maxiter
    
    for k in range(30):
        
        generation_number = 0
        cost_function_value = [initialCF]
        #we initialize the matrices that will hold the initial values
        population_1 = np.zeros((popsize,dimension))
        population_2 = np.zeros((popsize,dimension))
        #we start filling the matrice with initial values
        for i in range (popsize):
            for j in range (dimension):
                population_1[i][j] = bounds[0] + (bounds[1] - bounds[0]) * rand.random()
                
        while (generation_number < maxiter):
            
            for i in range(popsize):
                
                canidates = list(range(0,popsize))
                canidates.remove(i)
                random_index = sample(canidates, 3)
        
                x_1 = population_1[random_index[0]]
                x_2 = population_1[random_index[1]]
                x_3 = population_1[random_index[2]]
            
                crossover = rand.random()    
                
                
                for k in range (0, dimension):
                    if (crossover <= recombination):
                        population_2[i][k] = x_1[k] + mutate * (x_2[k] - x_3[k])
                    else:
                        population_2[i][k] = population_1[i][k]
                        
                if cost_func(population_2[i]) <= cost_func(population_1[i]):
                    population_1[i] = population_2[i]
                    
                    ensure_bounds(population_1[i], bounds, dimension)
                
            generation_number += 1
            cost_function_value.append(calculate_cost(population_1,cost_function_value[-1], cost_func, popsize))
            
        pyplot.plot(cost_function_value[1:])
        
        for i in range(0, len(cost_function_value)-1):
            average_cost_value[i] = (average_cost_value[i]*k + cost_function_value[i+1])/(k+1)
    
    pyplot.title('Convergence in the given number of iterations')
    pyplot.ylabel('Fitness (Cost Function)')
    pyplot.xlabel('Iterations')
    pyplot.show()
    
    pyplot.plot(average_cost_value)
    pyplot.title('Average evolution')
    pyplot.ylabel('Fitness(Cost Function)')
    pyplot.xlabel('Iterations')
    pyplot.show()
        
                        
def ensure_bounds(X, bounds, dimension):
    for k in range(dimension):
        #check if the value is over the upper boundary
        if X[k] > bounds[1]:
            X[k] = bounds[0] + rand.random()*(bounds[1] - bounds[0])
        
        #check if the value is below the lower boundary
        if X[k] < bounds[0]:
            X[k] = bounds[0] + rand.random()*(bounds[1] - bounds[0])
        
        #the value is between the boundaries
        else:
            X[k] = X[k]
        
# 1st deJong function:
def sphere(x):
    total = 0
    total = sum(x**2)
    return total

#2nd deJong function
def banana(x):
    total = 0
    for i in range(len(x)-2):
        total += 100 * ((x[i+1] - (x[i]**2))**2) + ((1 - (x[i]**2))**2)
    return total

# the Schwefel function
def schwefel(x):
    total = 0
    alpha = 418.982887
    for n in x:
        total = total + -n*math.sin(math.sqrt(math.fabs(n))) + alpha
    return total 

def calculate_cost(X, previous_value, cost_func, popsize):
    previous_cost = previous_value
    for j in range(0, popsize):
        cost = cost_func(X[j])
        if (previous_cost >= cost):
            previous_cost = cost
    return previous_cost
        
        
def main():
    cost_func = schwefel                  # Cost function
    upper_bound = 512
    lower_bound = -512        
    bounds = [lower_bound,upper_bound] # Bounds [(-512,512) for Schwefel, (-5.12, 5.12) for de Jong 1 and 2]
    popsize = 50                        # Population size, must be >= 4
    mutate = 0.5                        # Mutation factor 
    recombination = 0.9                # Recombination rate [0,1]
    maxiter = 100                      # Max number of generations (maxiter)
    dimension = 10                      # Number of dimensions
    initialCF = 2000
    minimize(cost_func, bounds, dimension, popsize, mutate, recombination, maxiter, initialCF)
    
    print("*******************************************************")
 
    cost_func = banana                  # Cost function
    upper_bound = 5.12
    lower_bound = -5.12        
    bounds = [lower_bound,upper_bound] # Bounds [(-512,512) for Schwefel, (-5.12, 5.12) for de Jong 1 and 2]
    popsize = 50                        # Population size, must be >= 4
    mutate = 0.5                        # Mutation factor 
    recombination = 0.9                # Recombination rate [0,1]
    maxiter = 100                     # Max number of generations (maxiter)
    dimension = 10                      # Number of dimensions
    initialCF = 1000
    minimize(cost_func, bounds, dimension, popsize, mutate, recombination, maxiter, initialCF)
    
    print("*******************************************************")
    
    cost_func = sphere                  # Cost function
    upper_bound = 5.12
    lower_bound = -5.12        
    bounds = [lower_bound,upper_bound] # Bounds [(-512,512) for Schwefel, (-5.12, 5.12) for de Jong 1 and 2]
    popsize = 50                        # Population size, must be >= 4
    mutate = 0.5                        # Mutation factor 
    recombination = 0.9                # Recombination rate [0,1]
    maxiter = 100                      # Max number of generations (maxiter)
    dimension = 10                      # Number of dimensions
    initialCF = 100
    minimize(cost_func, bounds, dimension, popsize, mutate, recombination, maxiter, initialCF)

