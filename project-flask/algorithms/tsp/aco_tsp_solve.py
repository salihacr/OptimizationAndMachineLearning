import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import math
import random
from math import radians, cos, sin, asin, sqrt
#
# Türkiye neresidir? 36° - 42° Kuzey enlemleri, 26°-45° Doğu boylamları arasında
#
# latitude = enlem 36 42 y ekseni
# longitude = boylam 26 45 x ekseni


class ACO_TSP_SOLVE(object):

    def __init__(self, _x_axis, _y_axis, _iteration=10, _ant_size=10, _rho=0.3, _alpha=1, _beta=1, _cities=[""]):
        self.x_axis = _x_axis
        self.y_axis = _y_axis
        self.iteration = _iteration
        self.ant_size = _ant_size  # len(self.x_axis)
        self.rho = _rho
        self.alpha = _alpha
        self.beta = _beta
        self.cities = _cities

        self.len_cities = len(_x_axis)
        # creating distance matrice from x and y distances
        self.distance_matrice = self.create_distance_matrice()

    global calculate_distance
    def calculate_distance(city1_x, city1_y, city2_x, city2_y):
        city1_x, city1_y, city2_x, city2_y = map(
            radians, [city1_x, city1_y, city2_x, city2_y])

        dlon = city2_x - city1_x
        dlat = city2_y - city1_y

        a = sin(dlat / 2) ** 2 + cos(city1_y) * \
            cos(city2_y) * sin(dlon / 2) ** 2
        dis = 2 * asin(sqrt(a)) * 6371 * 1000
        return dis / 1000.

    def create_distance_matrice(self):
        distance_matrice = np.zeros([len(self.x_axis), len(self.y_axis)])
        distance = 0.0
        for i in range(len(self.x_axis)):
            for j in range(len(self.y_axis)):
                distance = calculate_distance(
                    self.x_axis[i], self.y_axis[i], self.x_axis[j], self.y_axis[j])
                distance_matrice[i, j] = distance

                #print("[{0}][{1}] mesafe : {2} ".format(self.cities[i],self.cities[j],distance_matrice[i,j]))

        return distance_matrice
    global calculate_tour_cost
    def calculate_tour_cost(self, route):
        L = 0
        for i in range(len(route) - 1):
            L = L + self.distance_matrice[route[i], route[i+1]]

        return L

    global roulette_selection
    def roulette_selection(self, probability):
        cprobs = probability.copy()
        for i in range(1, len(probability)):
            cprobs[i] = cprobs[i-1] + probability[i]

        select_roulette = np.random.ranf()
        selection = np.where(select_roulette < cprobs)[0][0]

        return selection

    def run_optimize(self):

        distance = 1 / self.distance_matrice
        pheromone_amount = np.ones([self.len_cities, self.len_cities])
        Q = 1  # constant

        routes = np.zeros([self.ant_size, self.len_cities], dtype=int)

        cost = np.zeros(self.ant_size)

        best_cost = math.inf
        best_route = []

        cost_values = list()

        for t in range(self.iteration):
            for i in range(self.ant_size):

                visited_cities = list()
                visited_cities.append(np.random.randint(0, self.len_cities))

                routes[i, 0] = visited_cities[0]

                for j in range(1, self.len_cities):

                    end_city = visited_cities[-1]

                    probability = np.power(
                        pheromone_amount[end_city, :], self.alpha) * np.power(distance[end_city, :], self.beta)

                    probability[visited_cities] = 0

                    probability = probability / sum(probability)

                    selected_city = roulette_selection(self, probability)

                    routes[i, j] = selected_city

                    visited_cities.append(selected_city)

                    cost[i] = calculate_tour_cost(self, routes[i, :])

                if cost[i] < best_cost:
                    best_cost = cost[i]
                    best_route = routes[i, :]

                for i in range(self.ant_size):
                    route = routes[i, :]
                    route = np.append(route, route[0])

                # pheromone amount update
                for j in range(self.len_cities):
                    pos1 = route[j]
                    pos2 = route[j+1]
                    pheromone_amount[pos1,
                                     pos2] = pheromone_amount[pos1, pos2] + Q / cost[i]

            pheromone_amount = (1 - self.rho) * pheromone_amount
            cost_values.append(best_cost)

            #print("Iteration : {} , Best Cost : {} ".format(t+1, best_cost))

        return best_route, cost_values, best_cost

    global get_XY_location
    def get_XY_location(self, index):
        loc_xy = np.array([self.x_axis[index], self.y_axis[index]])
        return loc_xy

    def plot_cost_iteration(self, cost_values):
        fig, ax = plt.subplots(1, dpi=200)
        fig.suptitle('Ant Colony Optimization Costs per Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        ax.plot(cost_values, "r--", c="green", label = "Ant Colony")
        plt.legend()
        # plt.show()

        return fig

    def plot_cities(self, best_route, best_cost):

        cities_best_route = []

        for i in best_route:
            cities_best_route.append(self.cities[i])

        fig, ax = plt.subplots(1, figsize=(10, 6), dpi = 200)

        fig.suptitle('Ant Colony Optimization for TSP Problem')
        #plt.xlabel('X AXIS')
        #plt.ylabel('Y AXIS')

        ax.scatter(self.x_axis, self.y_axis, c="red", s=150)

        path = np.append(best_route, best_route)

        for i in range(len(self.cities)):
            ax.annotate(self.cities[i], xy=get_XY_location(self, i), c="black")

        plt.plot(self.x_axis[path], self.y_axis[path], c="green")
        # plt.show()

        #print("Best Route : {}, Best Cost : {} ".format(cities_best_route, best_cost))

        return fig