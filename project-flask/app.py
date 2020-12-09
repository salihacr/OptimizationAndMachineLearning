import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import os 
from io import BytesIO
import math, random, timeit

from helpers import helper

from algorithms.tsp.aco_tsp_solve import ACO_TSP_SOLVE
from algorithms.tsp.ga_tsp_solve import GA_TSP_SOLVE

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
"""
@app.route('/travellingsalesmanproblem')
def tsp():
    resimyol, onay = run()
    print("onay : ",onay)
    if onay == False:
        return render_template("tsp.html")
    else :
        print("onay : ",onay,"resim yol : ",resimyol)
        return render_template("tsp.html",resim_yol = resimyol, onayli = onay)
"""

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/travellingsalesmanproblem')
def tsp():
    return render_template("tsp.html")

def format_for_genetic(latitudes,longitudes):
    distance_list = zip(latitudes,longitudes)
    return list(distance_list)

@app.route('/travellingsalesmanproblem', methods = ["GET","POST"])
def run():

    file_type = "csv"
    if file_type == "text":
        df = pd.read_text("../input/marmara_mesafe/{}".format("mesafeler.txt")) 
    if file_type == "csv":
        df = pd.read_csv("data/marmara_mesafeler.csv")

    cities = df.loc[:, 'city'].values

    cities_x_axis = df.loc[:,'longitude'].values  # longitude is x axis

    cities_y_axis = df.loc[:,'latitude'].values # latitude is y axis

    show_label = False

    if request.method == "POST":      
        start = timeit.default_timer()
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

        cities_xy = format_for_genetic(cities_x_axis,cities_y_axis)
        
        aco_tsp = ACO_TSP_SOLVE(_x_axis = cities_x_axis,
                                _y_axis = cities_y_axis,
                                _iteration = int(iteration),
                                _ant_size = int(ant_size),
                                _rho = float(pheromone_rho),
                                _alpha = float(alpha),
                                _beta = float(beta),
                                _cities = cities)

        aco_best_route, aco_cost_values, aco_best_cost = aco_tsp.run_optimize()

        ga_tsp = GA_TSP_SOLVE(_cities_xy = cities_xy,
                                _cities = cities,
                                _life_count = int(life_count),
                                _cross_rate = float(cross_rate),
                                _mutation_rate = float(mutation_rate))

        ga_best_route, ga_cost_values, ga_best_cost  = ga_tsp.run(int(iteration))

        # bitti
        stop = timeit.default_timer()

        show_label = True

        # cities bird's-eye graphics
        #ga_tsp.plot_cities(cities_x_axis,cities_y_axis, bp[0], bp[2])
        #aco_tsp.plot_cities(best_route, best_cost)

        # cost iteration graphics
        #ga_tsp.plot_cost_iteration(bp[1])
        #aco_tsp.plot_cost_iteration(cost_values)

        # compare graphs via helper

        plt_compare_routes_fig = helper.compare_route_graphic(cities_x_axis, cities_y_axis, cities, aco_best_route, ga_best_route)
        
        plt_compare_costs_fig = helper.compare_cost_values(aco_cost_values, ga_cost_values)

        # save figures to upload via helper
        compare_route_fig_path = helper.save_figures_to_upload(plot_fig = plt_compare_routes_fig, img_name = "plt_compare_routes.png")

        compare_cost_fig_path = helper.save_figures_to_upload(plot_fig = plt_compare_costs_fig, img_name = "plt_compare_costs.png")
        
        #compare_route_fig_path = "static/uploads/karınca.png"
        print("onay {} resim yolu : {}".format(show_label, compare_route_fig_path))



        plt.style.use('ggplot')
        fig, ax = plt.subplots(1,dpi = 120)
        fig.suptitle('ACO vs GA Costs per Iterations')
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.grid(b = True, which = 'major', ls = '-.', lw = 0.45)
        plt.plot(aco_cost_values, c = "orange", label='Karınca')
        plt.plot(ga_cost_values,'-.',c = "#337ab7",label='Genetik')
        plt.legend(['Ant Colony', 'Genetic'])
        plt.legend()
        #plt.show()
        """
        canvas = FigureCanvas(fig)
        output = BytesIO()
        canvas.print_png(output)
        response = make_response(output.getvalue())
        response.mimetype = 'image/png'"""

        plt.savefig('uploads/aco_vs_ga.png')


        return render_template("tsp.html", resim_yol = compare_route_fig_path, onayli = show_label)
        #return compare_route_fig_path, show_label
    else:
        return render_template("tsp.html")
    #else:
        #return "",False

    #return render_template("tsp.html", show = show_label  ,route_compare_img = compare_route_fig_path)

    #return render_template("tsp.html", route_compare_img = compare_route_fig_path, cost_compare_img = compare_cost_fig_path)





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

@app.route('/machine_learning', methods = ["GET","POST"])
def machine_learning():
    return render_template("machine-learning.html")


@app.route("/hesapla", methods=['POST','GET'])
def hesapla():
    if request.method == 'POST':
        sayi = request.form.get('sayi') 
        sayi2 = request.form.get('sayi2') 
        sonuc = int(sayi)**2 + int(sayi2)**2
        print("çalıştım")
        return str(sonuc), str(sayi), str(sayi2)
    else:
        print("çalıştım")
        return "Bu sayfayı görmeye yetkiniz yok!"



###Using for Test
@app.route('/test', methods = ["GET","POST"])
def test():
    if request.method == "POST":
        sayi1 = request.form.get("sayi1")
        sayi2 = request.form.get("sayi2")
        sayi3 = request.form.get("sayi3")
        sayi4 = request.form.get("sayi4")

        result = toplama(sayi1,sayi2,sayi3,sayi4)

        return result

def toplama(sayi1, sayi2, sayi3, sayi4):
    total = int(sayi1) + int(sayi2)+ int(sayi3) + int(sayi4)
    print("Toplam = ", total)
    return total


if __name__ =="__main__":  
    app.run(debug = True)  