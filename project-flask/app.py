import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from opt_algorithms.tsp.acotspself import ACO_TSP
from opt_algorithms.tsp.genetictspself import *
import timeit

from flask import Flask,render_template,request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/optimization')
def optimization():
    return render_template("optimization.html")

@app.route('/travellingsalesmanproblem')
def tsp():
    runtsp()
    return render_template("tsp.html")

@app.route('/machine_learning')
def machine_learning():
    return render_template("machine-learning.html")

def tsp():
    pass
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
    
    acotsp = ACO_TSP( _longitude_x = boylam_x_ekseni,_latitude_y = enlem_y_ekseni, _city_list = cities, _iteration = 100)
    best_route, error_values, best_cost = acotsp.run()
    print("\nEn İyi Rota: ", best_route, "En İyi Rota Uzunluğu: ", best_cost)
    stop = timeit.default_timer()
    print('ACO İşlem Süresi: ', stop - start, " sn")


    start = timeit.default_timer()
    cityxylist = format_for_genetic(enlem_y_ekseni,boylam_x_ekseni)
    tsp = TSP(cityxylist)
    bp = tsp.main(100) #iterasyon sayisi

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

if __name__ =="__main__":  
    app.run(debug = True)  