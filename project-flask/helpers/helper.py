import os 
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from flask import make_response

plt.style.use('ggplot')

global get_XY_location
def get_XY_location(x_axis, y_axis, index):
    loc_xy = np.array([x_axis[index], y_axis[index]])
    return loc_xy 

def compare_cost_values(aco_cost_values, ga_cost_values):
    fig, ax = plt.subplots(1)
    fig.suptitle('ACO vs GA Costs per Iterations')
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(b = True, which = 'major', ls = '-.', lw = 0.45)
    plt.plot(aco_cost_values, c = "orange", label='Karınca')
    plt.plot(ga_cost_values,'-.',c = "#337ab7",label='Genetik')
    plt.legend(['Ant Colony', 'Genetic'])
    plt.legend()
    #plt.show()

    return fig

def compare_route_graphic( x_axis, y_axis, cities, aco_best_route, ga_best_route):
    
    fig, ax = plt.subplots(1, figsize=(12,8))

    fig.suptitle('ACO vs GA Optimization for TSP Problem')
    plt.xlabel('X AXIS')
    plt.ylabel('Y AXIS')

    # city points
    ax.scatter(x_axis, y_axis, c = "orange", s = 150)
    # and city names 
    for i in range(len(cities)):
        ax.annotate(cities[i] , xy = get_XY_location(x_axis, y_axis, i), c = "black")

    # aco path
    aco_path = np.append(aco_best_route, aco_best_route)

    # ga path
    ga_path = np.append(ga_best_route, ga_best_route)
    
    # adding paths to plot
    # aco path plot
    plt.plot(x_axis[aco_path], y_axis[aco_path], c = "purple")
    # ga path plot
    plt.plot(x_axis[ga_path], y_axis[ga_path], c = "green")

    #plt.show()

    return fig

def save_figures_to_upload(plot_fig, img_name):

    canvas = FigureCanvas(plot_fig)
    output = BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'

    #img_name +=

    plt.savefig('static/uploads/{}'.format(img_name))
    print("resim yolu : ", 'static/uploads/{}'.format(img_name))
    img_path = str('static/uploads/{}'.format(img_name))

    # uploadsa çevir

    return img_path

# using >  plt_img1_path = save_figures_to_upload(figure,your_img_name)