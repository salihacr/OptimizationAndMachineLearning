import numpy as np
import matplotlib.pyplot as plt

class DifferentialEvolution(object):
    def __init__(self, objective_function, bounds = [-5.12, 5.12], iteration = 10,
    problem_size = 4, population_size = 10, mutation_rate = 0.5, cr = 0.5): 

        self.objective_function = objective_function
        self.bounds = bounds
        self.iteration = iteration
        self.problem_size = problem_size

        self.mutation_rate = mutation_rate
        self.cr = cr
        self.population_size = population_size
        

    def run_optimize(self):
        population = np.random.ranf([self.population_size,self.problem_size]) * (self.bounds[1]-self.bounds[0]) + self.bounds[0]
        child = population.copy()
        solution = np.zeros(self.population_size)

        for i in range(self.population_size):
            solution[i]=self.objective_function(population[i,:])

        best_index = int(np.round(1+(self.population_size-1)*np.random.ranf()))
        
        cost_values = list()
        cost_values.append(solution[best_index])


        for iteration in range(self.iteration):
            for k in range(self.population_size):
                mutation_rate = np.random.permutation(self.population_size)[:3]
                mutation = self.mutation_rate*(population[mutation_rate[0], :] - \
                            population[mutation_rate[1], :]) + \
                            population[mutation_rate[2], :]
                
                for i in range(self.problem_size): 
                    if mutation[i] > self.bounds[1]:
                        mutation[i] = self.bounds[1]
                    elif mutation[i] < self.bounds[0]:
                        mutation[i] = self.bounds[0]

                for i in range(self.problem_size):
                    selection = np.random.ranf()
                    if selection > self.cr:
                        child[k, i] = population[k, i]
                    else:
                        child[k, i] = mutation[i]
                        
                child_solution = self.objective_function(child[k, :])
                if child_solution < solution[k]:
                    population[k,:] = child[k,:]
                    solution[k] = child_solution
                
                if min(solution) < solution[best_index]:
                    best_cost = min(solution)
                    idx = np.where(solution == best_cost)[0][0]
                    best_cost = self.objective_function(population[idx, :])
                
            cost_values.append(best_cost)
                
            print("Iteration :{}, Best Cost :{}".format(iteration + 1, best_cost))
        
        best_solve = population[idx, :]
        print("Optimum Karar değişkenleri :", best_solve)
        print("Optimum Çözüm :",best_cost)

        return cost_values, best_cost, best_solve

    def plot_results(self,cost_values):
        fig, ax = plt.subplots(1, dpi=200)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        ax.plot(cost_values, "r--", c="red", label = 'Differential Evolution')
        plt.legend()
        # plt.show()
        return fig

