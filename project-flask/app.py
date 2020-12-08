import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from opt_algorithms.tsp.acotspself import ACO_TSP
from opt_algorithms.tsp.genetictspself import *
from opt_algorithms.tsp.geneticNew import *
import timeit

from flask import Flask,render_template,request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/optimization')
def optimization():
    return render_template("genetik.html")

@app.route('/travellingsalesmanproblem')
def tsp():
    runtsp()
    return render_template("tsp.html")

###Using for Test
@app.route('/machine_learning', methods = ["GET","POST"])
def machine_learning():
    if request.method == "POST":
        sayi1 = request.form.get("sayi1")
        sayi2 = request.form.get("sayi2")
        sayi3 = request.form.get("sayi3")
        sayi4 = request.form.get("sayi4")

        toplama(sayi1,sayi2,sayi3,sayi4)
    return render_template("machine-learning.html")

def toplama(sayi1, sayi2, sayi3, sayi4):
    total = int(sayi1) + int(sayi2)+ int(sayi3) + int(sayi4)
    print("Toplam = ", total)

def format_for_genetic(latitudes,longitudes):
    distance_list = zip(latitudes,longitudes)
    return list(distance_list)

def runtsp():
    file_type = "csv"
    if file_type == "text":
        df = pd.read_text("../input/marmara_mesafe/{}".format("mesafeler.txt")) 
    if file_type == "csv":
        #df = pd.read_csv("project-flask\data\{}".format("marmara_mesafeler.csv"))
        df = pd.read_csv("data/marmara_mesafeler.csv")
    
    cities = df.iloc[:,0].values
    boylam_x_ekseni = df.iloc[:,1].values
    enlem_y_ekseni = df.iloc[:,2].values
        
    print("-----------------------------------")
    print("Ant Colony Optimization starting")
    start = timeit.default_timer()
    
    acotsp = ACO_TSP( _longitude_x = boylam_x_ekseni,_latitude_y = enlem_y_ekseni, _city_list = cities, _iteration = 10)
    best_route, error_values, best_cost = acotsp.run()
    print("\nEn İyi Rota: ", best_route, "En İyi Rota Uzunluğu: ", best_cost)
    stop = timeit.default_timer()
    print('ACO İşlem Süresi: ', stop - start, " sn")

    start = timeit.default_timer()
    cityxylist = format_for_genetic(enlem_y_ekseni,boylam_x_ekseni)
    tsp = TSP(cityxylist)
    bp = tsp.main(10) #iterasyon sayisi

    print("\nEn İyi Rota: ", bp[0], "En İyi Rota Uzunluğu: ", bp[2])
    stop = timeit.default_timer()
    print('GA İşlem Süresi: ', stop - start," sn")

    plt.style.use('ggplot')
    fig, ax = plt.subplots(1,dpi = 120)
    fig.suptitle('Costs per Iterations')
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(b = True, which = 'major', ls = '-.', lw = 0.45)
    plt.plot(error_values, c = "orange", label='Karınca')
    plt.plot(bp[1],'-.',c = "#337ab7",label='Genetik')
    plt.legend(['karınca', 'genetik'])
    plt.legend()
    plt.show()

    # karınca dolaşma grafiği
    #acotsp.plot_best_route_on_graph(best_route,best_cost)    
    
    # genetik dolaşma grafiği
    #     opt_algorithms.tsp.genetictspself.genetikcizim(bp[0],boylam_x_ekseni,enlem_y_ekseni,cities)


###Refactored GA Tsp ~ Test with paramaters    
@app.route('/genetik', methods = ["GET","POST"])
def genetikPage():
    if request.method == "POST":
        life_count = request.form.get("sayi1")
        cross_rate = request.form.get("sayi2")
        mutation_rate = request.form.get("sayi3")
        iterasyon = request.form.get("sayi4")

        genetikRun(life_count, cross_rate, mutation_rate, iterasyon)
    return render_template("genetik.html")


def genetikRun(life_count, cross_rate, mutation_rate, iterasyon):
    start = timeit.default_timer()

    file_type = "csv"
    if file_type == "text":
        df = pd.read_text("../input/marmara_mesafe/{}".format("mesafeler.txt")) 
    if file_type == "csv":
        df = pd.read_csv("data/marmara_mesafeler.csv")
    
    cities = df.iloc[:,0].values

    cities_y_axis = df.iloc[:,1].values

    cities_x_axis = df.iloc[:,2].values
        
    cities_xy = format_for_genetic(cities_x_axis,cities_y_axis)
    #print(cityxylist)

    """ (self,_cities_xy, _cities = [""], 
                _life_count = 100, _cross_rate = 0.4,
                _mutation_rate = 0.2, _gene_length = 10):
     """
    print(len(cities_xy))
    ga_tsp = GA_TSP_SOLVE(_cities_xy = cities_xy, _cities = cities)
   
    print("varsayılan değerler \n")
    # gen sayısı şehir sayısı kadar olmalıdır

    print("birey sayısı : ", ga_tsp.life_count)
    print("çaprazlama olasılığı : ", ga_tsp.cross_rate)
    print("mutasyon olasaılığı : ", ga_tsp.mutation_rate)
    print("gen sayısı : ", ga_tsp.gene_length)

    print("yeni değerlerden sonra \n")

    ga_tsp_yeni = GA_TSP_SOLVE(_cities_xy = cities_xy, _cities = cities,
                               _life_count = int(life_count),
                               _cross_rate = int(cross_rate),
                               _mutation_rate = int(mutation_rate))
   
    print("birey sayısı : ", ga_tsp_yeni.life_count)
    print("çaprazlama olasılığı : ", ga_tsp_yeni.cross_rate)
    print("mutasyon olasaılığı : ", ga_tsp_yeni.mutation_rate)
    print("gen sayısı : ", ga_tsp_yeni.gene_length)

    bp = ga_tsp.run(int(iterasyon)) #iterasyon sayisi

    print("\nEn İyi Rota: ", bp[0], "En İyi Rota Uzunluğu: ", bp[2])

    stop = timeit.default_timer()
    print('GA ~ TSP Hesap Süresi: ', stop - start, " sn")

if __name__ =="__main__":  
    app.run(debug = True)  