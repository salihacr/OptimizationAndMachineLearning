import numpy as np
import matplotlib.pyplot as plt

class SimulatedAnnealing(object):
    def __init__(self, objective_function, bounds = [-5.12, 5.12], iteration = 10,
    problem_size = 4, temperature = 10000, cooling_coefficient = 0.99, delta = 0.10):
        
        self.objective_function = objective_function
        self.bounds = bounds
        self.iteration = iteration * 50
        self.problem_size = problem_size

        self.temperature = temperature
        self.cooling_coefficient = cooling_coefficient
        self.delta = delta
        
    def run_optimize(self):

        solve = np.random.ranf([self.problem_size]) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

        solution_result = self.objective_function(solve)

        cost_values = list()
        best_solve = solve
        best_cost = solution_result

        bottom_change = (self.bounds[0] - self.bounds[1]) * self.delta / 2
        top_change = (self.bounds[1] - self.bounds[0]) * self.delta / 2
        
        temp = self.iteration / 50
        counter = 0
        end_temperature = 0.1
        while self.iteration > 0 and self.temperature > end_temperature :
            amount_of_change = np.random.ranf(self.problem_size) * (top_change - bottom_change) + bottom_change
            
            neighbour = solve + amount_of_change

            solution_result_neighbour = self.objective_function(neighbour)

            if solution_result_neighbour <= solution_result :

                solve = neighbour
                solution_result = solution_result_neighbour
            else :
                de = solution_result_neighbour - solution_result

                pa = np.exp(-de / self.temperature)

                roulette_selection = np.random.ranf()

                if roulette_selection < pa :
                    solve = neighbour
                    solution_result = solution_result_neighbour
            
            self.temperature = self.temperature * self.cooling_coefficient


            if self.iteration % 50 == 0 :
                cost_values.append(solution_result)

                if cost_values[counter] < best_cost :
                    best_cost = solution_result
                    best_solve = solve

                counter = counter + 1
                print("Iteration : {}, Best Cost : {}, Temperature : {}".format(counter, best_cost, self.temperature))
            
            self.iteration = self.iteration - 1

        remaining = int(temp) - counter
        print("remaining : ",remaining)
        for i in range(remaining):
            counter = counter + 1
            cost_values.append(best_cost)
            print("Iteration : {}, Best Cost : {}, Temperature : {}".format(counter, best_cost, self.temperature))

        return cost_values, best_cost, best_solve

    def plot_results(self,cost_values):
        fig, ax = plt.subplots(1, dpi=200)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        ax.plot(cost_values, "r--", c="blue", label = 'Simulated Annealing')
        plt.legend()
        # plt.show()
        return fig
