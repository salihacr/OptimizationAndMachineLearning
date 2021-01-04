import numpy as np
import math
import random
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt


# Şehirler Arası Mesafe Hesabı Yapılır
# İki şehrin kordinatları kullanılarak mesafeler hesaplanır.
def calculate_distance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])

    dlon = lng2 - lng1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    dis = 2 * asin(sqrt(a)) * 6371 * 1000
    return dis / 1000.


SCORE_NONE = -1
# birey


class Individual(object):
    def __init__(self, _gene=None):
        self.gene = _gene
        self.score = SCORE_NONE

# Genetik Algoritma


class GA(object):
    def __init__(self, _cross_rate, _mutation_rate, _life_count, _gene_length, _match_fun):
        self.cross_rate = _cross_rate  # Çaprazlama Olasılıgı
        self.mutation_rate = _mutation_rate  # Mutasyon Oranı
        self.life_count = _life_count  # Birey Sayısı
        self.gene_length = _gene_length  # Gen Uzunluğu
        self.match_fun = _match_fun  # Problem Boyutu

        self.lives = []  # Genler
        self.best_genes = None  # Nesildeki en iyi genler (korunacaklar)
        self.generation = 1  # Nesil Sayısı
        self.cross_count = 0  # Çaprazlama Sayısı
        self.mutation_count = 0  # Mutasyon Sayısı
        self.bounds = 0.0  # Uyum Fonk. İçin Bir Parametre
        self.mean = 1.0  # Ortalama Uyum Değeri
        self.init_population()

    def init_population(self):
        """İlk Nesil Oluşturulur"""
        self.lives = []
        for i in range(self.life_count):
            gene = [x for x in range(self.gene_length)]
            random.shuffle(gene)
            individual = Individual(gene)
            self.lives.append(individual)

    def judge(self):
        """Uyum Fonksiyonu Hesaplanır"""
        self.bounds = 0.0
        self.best_genes = self.lives[0]
        for individual in self.lives:
            individual.score = self.match_fun(individual)
            self.bounds += individual.score
            if self.best_genes.score < individual.score:
                self.best_genes = individual
        self.mean = self.bounds / self.life_count

    def cross(self, parent1, parent2):
        """Çaprazlama Yapılır"""
        n = 0
        while 1:
            new_genes = []
            # Belirlenen aralıkta rastgele tamsayı oluştur.
            index1 = random.randint(0, self.gene_length - 1)
            index2 = random.randint(index1, self.gene_length - 1)
            tempGene = parent2.gene[index1:index2]  # Çaprazlanan Genler
            len_parent1 = 0
            for g in parent1.gene:
                if len_parent1 == index1:
                    # Yeni gene çaprazlanan parçaları ekle.
                    new_genes.extend(tempGene)
                    len_parent1 += 1
                if g not in tempGene:
                    new_genes.append(g)
                    len_parent1 += 1
            if (self.match_fun(Individual(new_genes)) > max(self.match_fun(parent1), self.match_fun(parent2))):
                self.cross_count += 1
                return new_genes
            if (n > 100):
                self.cross_count += 1
                return new_genes
            n += 1

    def mutation(self, egg):
        """Mutasyon"""
        index1 = random.randint(0, self.gene_length - 1)
        index2 = random.randint(0, self.gene_length - 1)
        # Mutasyon sırasında ana popülasyonu etkilememek için yeni bir gen dizisi oluşturulur.
        new_genes = egg.gene[:]
        new_genes[index1], new_genes[index2] = new_genes[index2], new_genes[index1]
        if self.match_fun(Individual(new_genes)) > self.match_fun(egg):
            self.mutation_count += 1
            return new_genes
        else:
            rate = random.random()
            if rate < math.exp(-10 / math.sqrt(self.generation)):
                self.mutation_count += 1
                return new_genes
            return egg.gene

    def get_one(self):
        """Rulet Seçimi ike Rastgele Seçilim"""
        r = random.uniform(0, self.bounds)  # Rastgele seçilim.
        for individual in self.lives:
            r -= individual.score
            if r <= 0:
                return individual
        raise Exception("Hatalı Seçim", self.bounds)

    def new_child(self):
        """Yeni Çocuklar Üret"""
        parent1 = self.get_one()
        rate = random.random()
        # Belirtilen Oranda Çaprazlama Yapılır.
        if rate < self.cross_rate:
            # Çaprazlama işlemi.
            parent2 = self.get_one()
            gene = self.cross(parent1, parent2)
        else:
            gene = parent1.gene

            # Belirtilen olasılıkla mutasyon işlemi uygulanır.
        rate = random.random()
        if rate < self.mutation_rate:
            gene = self.mutation(Individual(gene))
        return Individual(gene)

    def next(self):
        """Yeni Nesil Üret"""
        self.judge()
        new_lives = []
        # En iyi bireyler korunarak gelecek nesillere aktarılır.
        new_lives.append(self.best_genes)
        while len(new_lives) < self.life_count:
            new_lives.append(self.new_child())
        self.lives = new_lives
        self.generation += 1

# Gezgin Satıcı Problemi


class GA_TSP_SOLVE(object):
    def __init__(self, _cities_xy, _cities=[""],
                 _life_count=100, _cross_rate=0.4,
                 _mutation_rate=0.2, _gene_length=11):

        self.init_cities(_cities_xy)
        self.cities = _cities
        # dışardan alınacak parametreler
        self.life_count = _life_count  # birey sayısı
        self.cross_rate = _cross_rate  # çaprazlama olasılığı
        self.mutation_rate = _mutation_rate  # mutasyon olasılığı
        self.gene_length = len(self.cities_xy)  # gen sayısı

        self.ga = GA(_life_count=self.life_count,
                     _cross_rate=self.cross_rate,
                     _mutation_rate=self.mutation_rate,
                     _gene_length=self.gene_length,
                     _match_fun=self.match_fun())

    def init_cities(self, _cities_xy):
        self.cities_xy = _cities_xy

    # Şehirler Arasındaki Mesafe Hesabı
    def distance(self, order):
        distance = 0.0
        for i in range(-1, len(self.cities_xy) - 1):
            index1, index2 = order[i], order[i + 1]
            #print("index1 {} - index2 {}".format(index1,index2))
            city1, city2 = self.cities_xy[index1], self.cities_xy[index2]
            distance += calculate_distance(city1[0],
                                           city1[1], city2[0], city2[1])

        return distance

    def match_fun(self):
        return lambda individual: 1.0 / self.distance(individual.gene)

    def run(self, n=0):
        cost_values = list()

        while n > 0:
            self.ga.next()
            cost = self.distance(self.ga.best_genes.gene)
            #print("{}. Generation, Cost : {}".format(self.ga.generation - 1, cost))
            self.ga.best_genes.gene.append(self.ga.best_genes.gene[0])
            best_route = self.ga.best_genes.gene
            cost_values.append(cost)
            #print("Güzergah: ", best_route)
            self.ga.best_genes.gene.pop()
            n -= 1

        best_cost = cost_values[-1]
        return best_route, cost_values, best_cost

    def plot_cost_iteration(self, cost_values):
        fig, ax = plt.subplots(1, dpi=200)
        fig.suptitle('Genetic Algorithm Costs per Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        ax.plot(cost_values,  "r--", c="blue", label = "Genetic Algorithm")
        plt.legend()
        # plt.show()

        return fig

    global get_XY_location

    def get_XY_location(self, x_axis, y_axis, index):
        loc_xy = np.array([x_axis[index], y_axis[index]])
        return loc_xy

    def plot_cities(self, x_axis, y_axis, best_route, best_cost):

        cities_best_route = []
        #print(self.cities)
        for i in best_route:
            cities_best_route.append(self.cities[i])

        fig, ax = plt.subplots(1, figsize=(10, 6), dpi = 200)

        fig.suptitle('Genetic Algorithm Optimization for TSP Problem')
        plt.xlabel('X AXIS')
        plt.ylabel('Y AXIS')

        ax.scatter(x_axis, y_axis, c="red", s=150)

        path = np.append(best_route, best_route)

        for i in range(len(self.cities)):
            ax.annotate(self.cities[i], xy=get_XY_location(
                self, x_axis, y_axis, i), c="black")

        plt.plot(x_axis[path], y_axis[path], c="blue")
        # plt.show()

        #print("Best Route : {}, Best Cost : {} ".format(cities_best_route, best_cost))

        return fig
