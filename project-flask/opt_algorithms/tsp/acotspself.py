import random
import math
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import numpy as np
#import warnings
#warnings.filterwarnings("ignore")

class ACO_TSP(object):
    def __init__(self,_latitude_y, _longitude_x, _iteration = 100,_ant_size = 11, _pheromone_evaporation_rate = 0.3,_alpha = 1, _beta = 1, _city_list = [""]):
        self.latitude_y = _latitude_y
        self.longitude_x = _longitude_x
        self.ant_size = _ant_size
        self.pheromone_evaporation_rate = _pheromone_evaporation_rate
        self.alpha = _alpha
        self.beta = _beta
        self.city_list = _city_list
        self.iteration = _iteration
        self.len_cities = len(_latitude_y)
        self.my_distance_matrice = self.distance_matrice_for_aco1()
    global geodistance       
    def geodistance(latitude1,longitude1,latitude2,longitude2):
        longitude1, latitude1, longitude2, latitude2 = map(radians, [longitude1, latitude1, longitude2, latitude2])
        dlon = longitude2 - longitude1
        dlat = latitude2 - latitude1
        a = sin(dlat / 2) ** 2 + cos(latitude1) * cos(latitude2) * sin(dlon / 2) ** 2
        dis = 2 * asin(sqrt(a)) * 6371 * 1000  
        return dis / 1000.
    
    def distance_matrice_for_aco1(self):
        dist_matrice = np.zeros([len(self.latitude_y),len(self.latitude_y)])
        singly_dist = 0.0
        for i in range(len(self.latitude_y)):
            for j in range(len(self.latitude_y)):
                singly_dist = geodistance(self.latitude_y[i],self.longitude_x[i],self.latitude_y[j],self.longitude_x[j])
                dist_matrice[i,j] = singly_dist
                #print("[{0}][{1}] mesafe : {2} ".format(self.city_list[i],self.city_list[j],dist_matrice[i,j]))
        self.my_distance_matrice = dist_matrice
        return dist_matrice
    
    global tour_cost
    def tour_cost(self,rota):
        L=0
        for i in range(len(rota)-1):
            L = L + self.my_distance_matrice[rota[i], rota[i+1]]
        return L
    
    global roulette
    def roulette(self,P):
        cprobs = P.copy()
        for i in range(1,len(P)):
            cprobs[i] = cprobs[i-1] + P[i]

        roulette_selection = np.random.ranf()
        selection = np.where(roulette_selection<cprobs)[0][0]
        return selection
    
    def run_optimization(self):
        
        temp_matrice = self.my_distance_matrice
        
        distance = 1 / temp_matrice
        
        pheromone_amount = np.ones([self.len_cities,self.len_cities])
        Q = 1

        routes = np.zeros([self.ant_size, self.len_cities],dtype=int)
        cost = np.zeros(self.ant_size)

        best_cost = math.inf
        best_route = []

        error_values = list()

        for t in range(self.iteration):
            for i in range(self.ant_size):
                visited_cities = list()
                visited_cities.append(np.random.randint(0,self.len_cities))
                routes[i,0] = visited_cities[0]

                for j in range(1,self.len_cities):
                    k = visited_cities[-1]
                    P = np.power(pheromone_amount[k,:],self.alpha) * np.power(distance[k,:],self.beta)
                    P[visited_cities] = 0
                    P=P/sum(P)
                    s = roulette(self,P)
                    routes[i,j] = s
                    visited_cities.append(s)

                cost[i] = tour_cost(self, routes[i,:])
                if cost[i] < best_cost:
                    best_cost = cost[i]
                    best_route = routes[i,:]

                for i in range(self.ant_size):
                    rota = routes[i,:]
                    rota = np.append(rota,rota[0])
                    for j in range(self.len_cities):
                        m = rota[j]
                        n = rota[j+1]
                        pheromone_amount[m,n]=pheromone_amount[m,n]+Q/cost[i]
            
            #print("en iyi rota tipi : {}".format(type(best_route)))
            pheromone_amount=(1 - self.pheromone_evaporation_rate) * pheromone_amount
            error_values.append(best_cost)
            print('Iteration : {}, Best Cost : {}'.format(t+1, best_cost))
            #print("en iyi rota : {}".format(best_route))
        
        temp_best_rota = np.array([0,1,2,3,4,5,6,7,8,9,10])
        #print("geçici rota tipi : {}".format(type(temp_best_rota)))
        #print("geçici en iyi rota : {}".format(temp_best_rota))
        temp_cost = tour_cost(self,rota = temp_best_rota)
        
        #print("geçici rota : {} maliyeti ise : {}".format(temp_best_rota,temp_cost))
        return best_route, error_values, best_cost
    
    global getXY
    def getXY(self,index):
        xy = np.array([self.latitude_y[index],self.longitude_x[index]])
        return xy
    
    def plot_best_route_on_graph(self,bestRoute, best_cost):        
        bestRouteOfCities = []
        for i in bestRoute:
            bestRouteOfCities.append(self.city_list[i])

        fig, ax = plt.subplots(1, dpi = 120, figsize = (12,8))
        fig.suptitle('Ant Colony Optimization for TSP Problem')
        plt.xlabel("Longitude BOYLAM X EKSENİ")
        plt.ylabel("Latitude ENLEM Y EKSENİ")
        

        #ax.scatter(self.longitude_x, self.latitude_y, c = "orange", s = 250)
        ax.scatter(self.longitude_x, self.latitude_y, c = "orange", s = 250)
        data = np.append(bestRoute, bestRoute)

        for i in range(len(self.city_list)):
            ax.annotate(self.city_list[i], xy = getXY(self, i), c = "purple", size = 12)
        
        plt.plot(self.latitude_y[data], self.longitude_x[data], c = "gray")
        plt.show();
        print("Best Route : {}, Best Cost : {} ".format(bestRouteOfCities,best_cost));
        
    def plot_cost_iteration_graph(self,error_values):
        fig, ax = plt.subplots(1,dpi = 120)
        fig.suptitle('Costs per Iterations')
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.grid(b = True, which = 'major', ls = '-.', lw = 0.45)
        ax.plot(error_values)
        plt.show()
        
    def run(self):
        best_route, error_values, best_cost = self.run_optimization()
        return best_route, error_values, best_cost