import numpy as np
import math
import matplotlib.pyplot as plt

#import warnings
#warnings.filterwarnings("ignore")

class ACO_TSP:
    def __init__(self, distance_matrice, len_cities,
                latitude, longitude, iteration = 100,
                ant_size = 30, pheromone_evaporation_rate = 0.3,
                alpha = 1, beta = 1, city_list = [""]):

        best_route, error_values, best_cost = run_optimization(distance_matrice,len_cities,iteration,ant_size,pheromone_evaporation_rate,alpha,beta)
        
        plot_best_route_on_graph(bestRoute = best_route, best_cost = best_cost, latitude = latitude, longitude = longitude, city_list = city_list)
        
        plot_cost_iteration_graph(error_values = error_values)
    
    global tour_cost
    def tour_cost(distance_matrice,rota):
        L=0
        for i in range(len(rota)-1):
            L = L + distance_matrice[rota[i], rota[i+1]]
        return L
    
    global roulette
    def roulette(P):
        cprobs = P.copy()
        for i in range(1,len(P)):
            cprobs[i] = cprobs[i-1] + P[i]

        roulette_selection = np.random.ranf()
        selection = np.where(roulette_selection<cprobs)[0][0]
        return selection
    
    global run_optimization
    def run_optimization(distance_matrice,len_cities,iteration,ant_size,pheromone_evaporation_rate,alpha,beta):

        distance = 1 / distance_matrice

        pheromone_amount = np.ones([len_cities,len_cities])
        Q = 1

        routes = np.zeros([ant_size, len_cities],dtype=int)
        cost = np.zeros(ant_size)

        best_cost = math.inf
        best_route = []

        error_values = list()

        for t in range(iteration):
            for i in range(ant_size):
                visited_cities = list()
                visited_cities.append(np.random.randint(0,len_cities))
                routes[i,0] = visited_cities[0]

                for j in range(1,len_cities):
                    k = visited_cities[-1]
                    P = np.power(pheromone_amount[k,:],alpha) * np.power(distance[k,:],beta)
                    P[visited_cities] = 0
                    P=P/sum(P)
                    s = roulette(P)
                    routes[i,j] = s
                    visited_cities.append(s)

                cost[i] = tour_cost(distance_matrice, routes[i,:])
                if cost[i] < best_cost:
                    best_cost = cost[i]
                    best_route = routes[i,:]

                for i in range(ant_size):
                    rota = routes[i,:]
                    rota = np.append(rota,rota[0])
                    for j in range(len_cities):
                        m = rota[j]
                        n = rota[j+1]
                        pheromone_amount[m,n]=pheromone_amount[m,n]+Q/cost[i]

            pheromone_amount=(1 - pheromone_evaporation_rate) * pheromone_amount
            error_values.append(best_cost)
            print('Iteration : {}, Best Cost : {}'.format(t+1, best_cost))
        return best_route, error_values, best_cost
    
    global getXY
    def getXY(x,y,index):
        xy = np.array([x[index],y[index]])
        return xy
    
    global plot_best_route_on_graph
    def plot_best_route_on_graph(bestRoute, best_cost, latitude, longitude, city_list):        
        bestRouteOfCities = []
        for i in bestRoute:
            bestRouteOfCities.append(city_list[i])

        fig, ax = plt.subplots(1, dpi = 120, figsize = (12,8))
        fig.suptitle('Ant Colony Optimization for TSP Problem')
        plt.xlabel("Latitude")
        plt.ylabel("Longitude")

        ax.scatter(latitude, longitude, c = "orange", s = 250)

        data = np.append(bestRoute, bestRoute)

        for i in range(len(city_list)):
            ax.annotate(city_list[i], xy = getXY(latitude, longitude, i), c = "purple", size = 12)
        
        plt.plot(latitude[data], longitude[data], c = "gray")
        plt.show();
        print("Best Route : {}, Best Cost : {} ".format(bestRouteOfCities,best_cost));
        
    global plot_cost_iteration_graph
    def plot_cost_iteration_graph(error_values):
        fig, ax = plt.subplots(1,dpi = 120)
        fig.suptitle('Costs per Iterations')
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.grid(b = True, which = 'major', ls = '-.', lw = 0.45)
        ax.plot(error_values)
        plt.show()