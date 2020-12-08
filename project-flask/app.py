import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import os 
from io import BytesIO
import math, random, timeit

from opt_algorithms.tsp.aco_tsp_solve import ACO_TSP_SOLVE
from opt_algorithms.tsp.ga_tsp_solve import GA_TSP_SOLVE

from flask import Flask, render_template, request, make_response


#import warnings 
#warnings.filterwarnings("ignore")

PEOPLE_FOLDER = os.path.join('project-flask','templates','static', 'img')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/optimization')
def optimization():
    return render_template("genetik.html")

@app.route('/travellingsalesmanproblem')
def tsp():
    return render_template("tsp.html")

def format_for_genetic(latitudes,longitudes):
    distance_list = zip(latitudes,longitudes)
    return list(distance_list)

@app.route('/tsp', methods = ["GET","POST"])
def run():

    file_type = "csv"
    if file_type == "text":
        df = pd.read_text("../input/marmara_mesafe/{}".format("mesafeler.txt")) 
    if file_type == "csv":
        df = pd.read_csv("data/marmara_mesafeler.csv")

    cities = df.loc[:, 'city'].values

    cities_x_axis = df.loc[:,'longitude'].values  # longitude is x axis

    cities_y_axis = df.loc[:,'latitude'].values # latitude is y axis

    print(cities)
    print(cities_x_axis)
    print(cities_y_axis)

    if request.method == "POST":
        print("çalıştım")
        
        # karınca parametreleri
        ant_size = request.form.get("ant_size")
        pheromone_rho = request.form.get("pheromone_rho")
        alpha = request.form.get("alpha")
        beta = request.form.get("beta")

        # genetik parametreleri
        life_count = request.form.get("life_count")
        mutation_rate = request.form.get("mutation_rate")
        cross_rate = request.form.get("cross_rate")

        # iterasyon
        iteration = request.form.get("iteration")

    #(self, _x_axis, _y_axis, _iteration = 10, _ant_size = 10, _rho = 0.3, _alpha = 1, _beta = 1, _cities = [""]):
    
    cities_xy = format_for_genetic(cities_x_axis,cities_y_axis)
    print(cities_xy)
    
    aco_tsp = ACO_TSP_SOLVE(_x_axis = cities_x_axis,
                            _y_axis = cities_y_axis,
                            _iteration = int(iteration),
                            _ant_size = int(ant_size),
                            _rho = float(pheromone_rho),
                            _alpha = float(alpha),
                            _beta = float(beta),
                            _cities = cities)

    best_route, cost_values, best_cost = aco_tsp.run_optimize()

    ga_tsp = GA_TSP_SOLVE(_cities_xy = cities_xy,
                               _cities = cities,
                               _life_count = int(life_count),
                               _cross_rate = float(cross_rate),
                               _mutation_rate = float(mutation_rate))

    bp = ga_tsp.run(int(iteration))

    # cities bird's-eye graphics
    #ga_tsp.plot_cities(cities_x_axis,cities_y_axis, bp[0], bp[2])
    #aco_tsp.plot_cities(best_route, best_cost)

    # cost iteration graphics
    #ga_tsp.plot_cost_iteration(bp[1])
    #aco_tsp.plot_cost_iteration(cost_values)

    plt.style.use('ggplot')
    fig, ax = plt.subplots(1,dpi = 120)
    fig.suptitle('ACO vs GA Costs per Iterations')
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(b = True, which = 'major', ls = '-.', lw = 0.45)
    plt.plot(cost_values, c = "orange", label='Karınca')
    plt.plot(bp[1],'-.',c = "#337ab7",label='Genetik')
    plt.legend(['Ant Colony', 'Genetic'])
    plt.legend()
    plt.show()

    canvas = FigureCanvas(fig)
    output = BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'

    plt.savefig('uploads/aco_vs_ga.png')

    #full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'compare.png')
    
    full_filename = "data/aco_vs_ga.png"

    #project-flask\templates\static\img\
    print("isim : ", full_filename)


    return render_template("tsp.html", compare_image = full_filename)































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


if __name__ =="__main__":  
    app.run(debug = True)  