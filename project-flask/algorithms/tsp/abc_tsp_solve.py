# https://github.com/Kivanc10/Artifical-Bee-Colony-Algorithms/tree/master/beeColonies(TSP) original code

import math
from math import radians, cos, sin, asin, sqrt
from random import uniform,randint,sample # liste karıştırmak için sample,değer üretmek için randint ve uniform

import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt

class ABC_TSP_SOLVE(object):

    def __init__(self, x_axis, y_axis, iteration = 10, number_of_bees = 10, number_of_worker_bees = 5, limits_of_try = 5, cities = [""]):
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.iteration = iteration
        self.number_of_bees = number_of_bees
        self.number_of_feeds = number_of_bees
        self.number_of_worker_bees = number_of_worker_bees
        self.limits_of_try = limits_of_try
        self.cities = cities
        self.distance_matrix = self.create_distance_matrice()

    global calculate_distance
    def calculate_distance(city1_x, city1_y, city2_x, city2_y):
        city1_x, city1_y, city2_x, city2_y = map(radians, [city1_x, city1_y, city2_x, city2_y])
        
        dlon = city2_x - city1_x
        dlat = city2_y - city1_y

        a = sin(dlat / 2) ** 2 + cos(city1_y) * cos(city2_y) * sin(dlon / 2) ** 2
        dis = 2 * asin(sqrt(a)) * 6371 * 1000  
        return dis / 1000.

    def create_distance_matrice(self):
        distance_matrix = np.zeros([len(self.x_axis), len(self.y_axis)])
        distance = 0.0
        for i in range(len(self.x_axis)):
            for j in range(len(self.y_axis)):
                distance = calculate_distance(self.x_axis[i], self.y_axis[i], self.x_axis[j], self.y_axis[j])
                distance_matrix[i, j] = distance
                
                #print("[{0}][{1}] mesafe : {2} ".format(self.cities[i], self.cities[j], distance_matrix[i, j]))
        return distance_matrix

    global calculate_cost
    def calculate_cost(self, solution): # mesafe hesaplayan fonksiyon(calculate distance function)
        total_distance = 0 # maliyet hesaplayan fonk
        index = solution[0]
        for next_index in solution[1:]:
            total_distance +=  self.distance_matrix[index][next_index]
            index = next_index
        return total_distance # return of total_distance

    global swap
    def swap(self, sequence, i, j): # indis değiştirme fonskyonu(change index of list)
        temp = sequence[i]
        sequence[i] = sequence[j] #indis değiştirme fonk
        sequence[j] = temp

    global randF
    def randF(): # 0 ile 1 arasında rastgele değer üreten fonksiyon.(randomly protect number between 0-1)
        return uniform(0.0001, 0.9999)
    
    global goMethods    
    goMethods = int(2 + ((randF()-0.5) * 2) * (2.5 - 1.2)) # karıncalar için ilerleme metodu.(method of ants move)

    global roulette_selection
    def roulette_selection(self, bees): # gözcü arıları seçmek için rulet yöntemi(rouletta function to selected to obserer bee)
        total = 0
        section = 0
        for i in range(len(bees)):
            total += (1/float(bees[i][1])) # toplam uygunluk değeri hesaplama(calculate total fitness value)
        probability = [] # olasılık listesi
        for i in range(len(bees)):
            section += float((1/int(bees[i][1]))/total) # herbirinin seçilme olasılığı(toplam uygunluk değerine bölerek bulunur.)(calculate probability of by selected)
            probability.append(section) # olasılık listesine eklendi.(append list of probability that every probability)
        next_generation = [] # yeni jenerasyon
        for i in range(self.number_of_feeds): # besin sayısı kadar (gözcü arı sayısına eşit) seçim yapma ve seçme( until num of feeds(same time num of equal observer bee))
            choice = randF() # random seçimler yapılıyor.
            for j in range(len(probability)): # range inside of probability
                if (choice <=  probability[j]):
                    next_generation.append(bees[j])
                    break
        temp = sample(next_generation, self.number_of_feeds) # yeni jenerasyonun içi karıştırıldı ve fonksiyonda döndürüldü
        next_generation = temp
        return next_generation
    
    global approach_to_good
    def approach_to_good(self, bees, a, b, c): # çözüm kümelerine uygulanan iyileştirme formülü(improvement to approach good path)
        bees = bees[0][:]
        swap(self, bees, a, b)
        swap(self, bees, b, c)
        return (bees, calculate_cost(self,bees))

    def remove_bees(self, bees): # rota ortadan kaldırma (eleme) fonksiyonu deneme limitine bağlıdır.(elimination of path and bees)(that is necessary for algorithm, forget path)
        bees = bees[0][:]
        index1 = randint(0, city_count-1)
        index2 = randint(0, city_count-1)
        index3 = randint(0, city_count-1)
        index4 = randint(0, city_count-1)
        
        swap(self, bees, index1, index2)
        swap(self, bees, index2, index3)
        swap(self, bees, index3, index4)
    
        return (bees, calculate_cost(self, bees)) # this operation do via multi-swap.To sum up provides to eliminate path and bee groups.
    
    def start_optimize(self):
        # description values that necessary of algorithm

        city_count = len(self.distance_matrix)  #GSP büyüklüğü
        bees = [] # arı için array
        first_path = list(range(0, city_count)) # çizilin ilk yol(rota) , initialize path(begin)
        index = 0

        cost_values = list()

        for i in range(self.number_of_bees): # başlangıçta arı sayısı kadar rota oluşturuldu.(in the begin, occur paths until number of bees)
            path = sample(first_path, city_count)
            bees.append((path, calculate_cost(self, path)))
        bees.sort(key = lambda x:x[1]) # en az maliyete göre sıraladık.(sort of the most least cost)


                
        for iter in range(self.iteration): # beginnig of self.iteration

            count = 0 # sayaç , counter
            best_bees = bees[randint(0, goMethods)] # metotla yola çıkacak grup belirlenir.(determine the move group via goMethods and random methods)
            for j in range(0, city_count):
                morePowerBees = approach_to_good(self, best_bees, randint(0, city_count-1), randint(0, city_count-1), randint(0, city_count-1)) # gidecek gruba iyileştirme uygulanır.(apply improvemention to move group)
                if (bees[j][1]>morePowerBees[1]): # Elde edilen çözüm değeri önceki çözüm değerinden daha iyi ise listeye alınır.(change value to least cost to high cost)
                    bees[j] = morePowerBees # listeye alındı.
                else:
                    self.limits_of_try += 1 # eğer iyileşme sağlanırsa limit değeri 1 arttırılır.(if not necessary improvement, increase limits of array)
            bees.sort(key = lambda x:x[1]) # en az maliyete göre sıralanır.
            for i in range(self.number_of_bees-self.number_of_worker_bees, self.number_of_bees):
                observer = roulette_selection(self, bees) # gözcü arılar rulet seçimine göre seçilir.
                for l in range(self.number_of_worker_bees, self.number_of_bees):
                    bees[l] = observer[count] # gözcü arılardan az maliyetli olanlar arı listesine alınır.
            count += 1
            if (count>self.limits_of_try): # eğer sayaç deneme limitinden fazla ise arılar ortadan kaldırılır.(remove_bees fonskiyonu ile)(if counter bigger than limits of array, then apply last operation that remove_bees)
                for k in range(city_count):
                    bees[k] = remove_bees(bees[k]) # iyileştirilemeyen çözüm kümeleri(rotalar) kaldırıldı.
            bees.sort(key = lambda x:x[1])  # en az maliyete göre sırala.
            
            best_cost = bees[0][1]
            cost_values.append(best_cost)
            best_route = bees[0][0]

            #print("Iteration : {} , Best Cost : {} ".format(iter + 1, best_cost))

        return best_route, cost_values, best_cost

    global get_XY_location
    def get_XY_location(self,index):
        loc_xy = np.array([self.x_axis[index], self.y_axis[index]])
        return loc_xy 
    
    def plot_cost_iteration(self, cost_values):
        fig, ax = plt.subplots(1, dpi = 120)
        fig.suptitle('Costs per Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Cost') 
        ax.plot(cost_values,'r--', label = "Artificial Bee Colony", c= "orange")
        plt.legend()
        return fig
    
    def plot_cities(self, best_route, best_cost):
        
        cities_best_route = []
        
        for i in best_route:
            cities_best_route.append(self.cities[i])
        
        fig, ax = plt.subplots(1, figsize=(10, 6), dpi = 200)

        fig.suptitle('Artificial Bee Colony Optimization for TSP Problem')
        plt.xlabel('X AXIS')
        plt.ylabel('Y AXIS')

        ax.scatter(self.x_axis, self.y_axis, c = "red", s = 150)

        path = np.append(best_route, best_route)

        for i in range(len(self.cities)):
            ax.annotate(self.cities[i] , xy = get_XY_location(self, i), c = "black")
        
        plt.plot(self.x_axis[path], self.y_axis[path], c = "green", label = "Artificial Bee Colony")
        plt.legend()

        #print("Best Route : {}".format(cities_best_route))
        #print("Best Cost : {} ".format(best_cost))

        return fig