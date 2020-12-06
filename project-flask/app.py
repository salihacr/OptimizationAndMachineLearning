import numpy as np
import math
import matplotlib.pyplot as plt
from opt_algorithms.tsp.acotsp import ACO_TSP

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
    #runtsp()
    return render_template("tsp.html")

@app.route('/machine_learning')
def machine_learning():
    return render_template("machine-learning.html")

def runtsp():
    # MARMARA BÖLGESİ İLLER X Y KOORDİNATLARI
    x = np.array([26.562269, 27.216667, 27.516667,
                28.97696, 29.88152, 30.435763, 29.266667,
                29.063448, 30.066524, 27.88261, 26.41416])

    y = np.array([41.681808,41.733333,40.983333,
                41.00527,40.85327,40.693997,40.65,
                40.266864, 40.056656, 39.648369, 40.155312])
    len_cities = len(x)
    distances = np.array([
                        [0,65.4,143,237,341,394,330,391,438,518,222],
                        [66.1,0,117,212,315,368,305,365,412,492,240],
                        [144,118,0,146,250,310,240,300,347,190,194],
                        [239,213,146,0,97.1,157,93.4,154,201,281,342],
                        [334,308,247,97.1,0,75.8,77.5,138,164,265,392],
                        [395,369,308,157,70.2,0,125,185,107,312,439],
                        [331,305,238,94.7,83.2,125,0,71.0,111,198,325],
                        [392,366,299,155,144,185,68.5,0,97.1,147,274],
                        [437,411,344,201,158,107,110,96.1,0,243,370],
                        [519,493,192,282,271,312,198,148,245,0,194],
                        [223,241,195,341,398,440,326,275,372,194,0]
                        ])
    # Şehir lokasyonlarının x ve y koordinatları
    liste = ["Edirne","Kırklareli","Tekirdağ",
            "İstanbul","Kocaeli","Sakarya","Yalova",
            "Bursa","Bilecik","Balıkesir","Çanakkale"]

    acotsp = ACO_TSP(distance_matrice = distances, len_cities = len_cities, latitude = x, longitude = y, city_list = liste, iteration = 100);

if __name__ =="__main__":  
    app.run(debug = True)  