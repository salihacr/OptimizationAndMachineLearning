import numpy as np
import matplotlib.pyplot as plt

class PSO(object):
    def __init__(self, objective_function, bounds = [-5.12, 5.12], iteration = 10,
    problem_size = 4, particle_size = 10, w = 0.8):

        self.objective_function = objective_function
        self.bounds = bounds
        self.iteration = iteration
        self.problem_size = problem_size
        self.particle_size = particle_size

        self.w = w

    def run_optimize(self):
        c1 = 2
        c2 = 2

        population = np.random.ranf([self.particle_size, self.problem_size]) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

        solution = np.zeros(self.particle_size)

        for i in range(self.particle_size):
            solution[i] = self.objective_function(population[i,:])
        
        velocity = np.zeros([self.particle_size, self.problem_size])
        
        p_best_value = solution
        p_best_position = population
        

        g_best_value = min(solution)
        index = np.where(solution == g_best_value)
        g_best_position = population[index, :]

        cost_values = list()

        cost_values.append(g_best_value)

        for k in range(self.iteration):
            for i in range(self.particle_size):
                velocity[i, :] = self.w * velocity[i, :] + \
                    c1 * np.random.ranf() * (p_best_position[i, :] - population[i, :]) + \
                        c2 * np.random.ranf() * (g_best_position - population[i, :])
            vmax = (self.bounds[1] - self.bounds[0]) / 2
            
            for i in range(self.particle_size):
                for j in range(self.problem_size):
                    if velocity[i, j] > vmax :
                        velocity[i, j] = vmax
                    elif velocity[i, j] < -vmax:
                        velocity[i, j] = -vmax

            population = population + velocity

            for i in range(self.particle_size):
                for j in range(self.problem_size):
                    if population[i, j] > self.bounds[1]:
                        population[i, j] = self.bounds[1]
                    elif population[i, j] < self.bounds[0]:
                        population[i, j] = self.bounds[0]

            for i in range(self.particle_size):
                solution[i]=self.objective_function(population[i,:])

            for i in range(self.particle_size):
                if solution[i] < p_best_value[i]:
                    p_best_value[i, :] = population[i, :]
                    p_best_value[i] = solution[i]

            if min(solution) < g_best_value:
                g_best_value = min(solution)
                idx = np.where(solution == g_best_value)
                g_best_position = population[idx, :]
            
            cost_values.append(g_best_value)
            
            #print("Iteration :{}, Best Cost :{}".format(k + 1, g_best_value))
        
        return cost_values, g_best_value

    def plot_results(self,cost_values):
        fig, ax = plt.subplots(1, figsize=(10, 6), dpi=200)
        fig.suptitle('Particle Swarm Optimization')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        ax.plot(cost_values, "r--", c="green", label = 'Particle Swarm')
        plt.legend()
        # plt.show()
        return fig