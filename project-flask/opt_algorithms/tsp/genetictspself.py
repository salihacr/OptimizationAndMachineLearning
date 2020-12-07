import random
import math
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt


#Şehirler Arası Mesafe Hesabı Yapılır 
#İki şehrin kordinatları kullanılarak mesafeler hesaplanır. 
def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    
    dlon = lng2 - lng1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    dis = 2 * asin(sqrt(a)) * 6371 * 1000  
    return dis / 1000.

SCORE_NONE = -1
#Gen
class Life(object):
    def __init__(self, aGene=None):
        self.gene = aGene
        self.score = SCORE_NONE
#Genetik Algoritma 
class GA(object):
    def __init__(self, aCrossRate, aMutationRage, aLifeCount, aGeneLenght, aMatchFun):
        self.croessRate = aCrossRate  # Çaprazlama Olasılıgı
        self.mutationRate = aMutationRage  #Mutasyon Oranı
        self.lifeCount = aLifeCount  #Birey Sayısı
        self.geneLenght = aGeneLenght  #Gen Uzunluğu
        self.matchFun = aMatchFun  #Problem Boyutu
        self.lives = []  #Genler
        self.best = None  #Nesildeki en iyi genler (korunacaklar)
        self.generation = 1  #Nesil Sayısı
        self.crossCount = 0  #Çaprazlama Sayısı
        self.mutationCount = 0  #Mutasyon Sayısı
        self.bounds = 0.0  #Uyum Fonk. İçin Bir Parametre
        self.mean = 1.0  #Ortalama Uyum Değeri
        self.initPopulation()

    def initPopulation(self):
        """İlk Nesil Oluşturulur"""
        self.lives = []
        for i in range(self.lifeCount):
            gene = [x for x in range(self.geneLenght)]
            random.shuffle(gene)  # 用来对一个元素序列进行重新随机排序
            life = Life(gene)
            self.lives.append(life)

    def judge(self):
        """Uyum Fonksiyonu Hesaplanır"""
        self.bounds = 0.0
        self.best = self.lives[0]
        for life in self.lives:
            life.score = self.matchFun(life)
            self.bounds += life.score
            if self.best.score < life.score:
                self.best = life
        self.mean = self.bounds / self.lifeCount

    def cross(self, parent1, parent2):
        """Çaprazlama Yapılır"""
        n = 0
        while 1:
            newGene = []
            index1 = random.randint(0, self.geneLenght - 1)  #Belirlenen aralıkta rastgele tamsayı oluştur.
            index2 = random.randint(index1, self.geneLenght - 1)
            tempGene = parent2.gene[index1:index2]  #Çaprazlanan Genler
            p1len = 0
            for g in parent1.gene:
                if p1len == index1:
                    newGene.extend(tempGene)  #Yeni gene çaprazlanan parçaları ekle.
                    p1len += 1
                if g not in tempGene:
                    newGene.append(g)
                    p1len += 1
            if (self.matchFun(Life(newGene)) > max(self.matchFun(parent1), self.matchFun(parent2))):
                self.crossCount += 1
                return newGene
            if (n > 100):
                self.crossCount += 1
                return newGene
            n += 1

    def mutation(self, egg):
        """Mutasyon"""
        index1 = random.randint(0, self.geneLenght - 1)
        index2 = random.randint(0, self.geneLenght - 1)
        newGene = egg.gene[:]  #Mutasyon sırasında ana popülasyonu etkilememek için yeni bir gen dizisi oluşturulur.
        newGene[index1], newGene[index2] = newGene[index2], newGene[index1]
        if self.matchFun(Life(newGene)) > self.matchFun(egg):
            self.mutationCount += 1
            return newGene
        else:
            rate = random.random()
            if rate < math.exp(-10 / math.sqrt(self.generation)):
                self.mutationCount += 1
                return newGene
            return egg.gene

    def getOne(self):
        """Rulet Seçimi ike Rastgele Seçilim"""
        r = random.uniform(0, self.bounds)  #Rastgele seçilim.
        for life in self.lives:
            r -= life.score
            if r <= 0:
                return life  
        raise Exception("Hatalı Seçim", self.bounds)

    def newChild(self):
        """Yeni Çocuklar Üret"""
        parent1 = self.getOne()
        rate = random.random()
        #Belirtilen Oranda Çaprazlama Yapılır.
        if rate < self.croessRate:
            #Çaprazlama işlemi.
            parent2 = self.getOne()
            gene = self.cross(parent1, parent2)
        else:
            gene = parent1.gene

            #Belirtilen olasılıkla mutasyon işlemi uygulanır.
        rate = random.random()
        if rate < self.mutationRate:
            gene = self.mutation(Life(gene))
        return Life(gene)

    def next(self):
        """Yeni Nesil Üret"""
        self.judge()
        newLives = []
        newLives.append(self.best)  #En iyi bireyler korunarak gelecek nesillere aktarılır.
        while len(newLives) < self.lifeCount:
            newLives.append(self.newChild())
        self.lives = newLives
        self.generation += 1

#Gezgin Satıcı Problemi
class TSP(object):
    def __init__(self,cities,aLifeCount = 100 ):
        self.initCitys(cities)
        self.lifeCount = aLifeCount
        self.ga = GA(aCrossRate=0.4,
                     aMutationRage=0.2,
                     aLifeCount=self.lifeCount,
                     aGeneLenght=len(self.citys),
                     aMatchFun=self.matchFun())

    def initCitys(self,cities):
        #Marmara Bölgesindeki 11 İlimizin Kordinatları
        self.citys = cities
        
    #Şehirler Arasındaki Mesafe Hesabı
    def distance(self, order):
        distance = 0.0
        for i in range(-1, len(self.citys) - 1):
            index1, index2 = order[i], order[i + 1]
            city1, city2 = self.citys[index1], self.citys[index2]
            distance += geodistance(city1[0], city1[1], city2[0], city2[1])
            #print("city1[0]",city1[0],"city1[1]",city1[1],"city2[0]",city2[0],"city2[1]",city2[1])
        return distance

    def matchFun(self):
        return lambda life: 1.0 / self.distance(life.gene)

    def main(self, n=0):
        
        error_values = list()
        
        while n > 0:
            self.ga.next()
            distance = self.distance(self.ga.best.gene)     
            print(("Nesil: %4d \t\t Mesafe: %f") % (self.ga.generation - 1, distance))  #Nesil sayısı ve rota uzunluğu.
            self.ga.best.gene.append(self.ga.best.gene[0])
            bestPath = self.ga.best.gene
            error_values.append(distance)
            print("Güzergah: ", bestPath)  #Sırayla gezilen şehirler.
            self.ga.best.gene.pop()        
            n -= 1
        return bestPath, error_values, distance
           
#Şehirler arasında yol çizim grafiği
def draw(bestPath, cities):
    ax = plt.subplot(111, aspect='equal')
    x = []
    y = []
    for i in range(-1, len(cities) - 1):
        index = bestPath[i]
        city = cities[index]
        x.append(city[0])
        y.append(city[1])
    x.append(x[0])
    y.append(y[0])
    ax.plot(x, y)
    plt.show()
    
    
#Grafik
def plot_cost_iteration_graph(error_values):
    fig, ax = plt.subplots(1,dpi = 120)
    fig.suptitle('Costs per Iterations')
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(b = True, which = 'major', ls = '-.', lw = 0.45)
    ax.plot(error_values)
    plt.show()

def getXY2(longitude,latitude,index):
    xy = np.array([longitude[index],latitude[index]])
    return xy

def genetikcizim(bestRoute, lonX,latY,city_list):        
    bestRouteOfCities = []
    for i in bestRoute:
        bestRouteOfCities.append(city_list[i])

    fig, ax = plt.subplots(1, dpi = 120, figsize = (12,8))
    fig.suptitle('Ant Colony Optimization for TSP Problem')
    plt.xlabel("Longitude BOYLAM X EKSENİ")
    plt.ylabel("Latitude ENLEM Y EKSENİ")


    ax.scatter(lonX, latY, c = "orange", s = 250)

    data = np.append(bestRoute, bestRoute)

    for i in range(len(city_list)):
        ax.annotate(city_list[i], xy = getXY2(lonX,latY,i), c = "purple", size = 12)

    plt.plot(lonX[data], latY[data], c = "gray")
    plt.show();
    print("Best Route : {}, Best Cost : {} ".format(bestRouteOfCities,best_cost))