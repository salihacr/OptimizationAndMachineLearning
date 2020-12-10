import os 
from io import BytesIO

import random, string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from apscheduler.schedulers.background import BackgroundScheduler
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
    plt.plot(x_axis[aco_path], y_axis[aco_path], c = "purple", label='Karınca')
    # ga path plot
    plt.plot(x_axis[ga_path], y_axis[ga_path],"-_", c = "green", label='Genetik')

    plt.legend(['Ant Colony', 'Genetic'])

    #plt.show()

    return fig


def save_figures_to_upload(plot_fig, img_name):
    # random + img_name
    #rand_str = get_random_string(10)
    #temp_img_name = rand_str + img_name

    # pure img_name
    temp_img_name = img_name

    base_path = 'static/uploads/'
    img_path = base_path + temp_img_name

    plt.savefig(img_path)
    print("Image Path => ", img_path)
    
    return img_path

# using >  plt_img1_path = save_figures_to_upload(figure,your_img_name)

#Random str generator for image names
def get_random_string(length):
    letters = string.ascii_letters
    value = random.randint(100, 999)
    result_str = ''.join(random.choice(letters) for i in range(length))
    result_str += str(value)
    print("Random string of length", length, "is:", result_str)
    return result_str

def format_for_genetic(longitudes_x,latitudes_y):
    distance_list = zip(longitudes_x,latitudes_y)
    return list(distance_list)


#Delete Folder Şedule
import schedule, time, os, glob
def deleteFileInFolder():
    dir = 'static/uploads'
    filelist = glob.glob(os.path.join(dir, "*"))
    if len(filelist) != 0:
        for f in filelist:
            os.remove(f)
 
def runSchedule():
    schedule.every(.5).minutes.do(deleteFileInFolder)
    #schedule.every(2).hour.do(deleteFileInFolder)

    while True:
        schedule.run_pending()
        time.sleep(1)




def testSalih():
    def deleteFileInFolder():
        dir = 'static/uploads'
        filelist = glob.glob(os.path.join(dir, "*"))
        if len(filelist) != 0:
            for f in filelist:
                os.remove(f)

    sched = BackgroundScheduler(daemon=True)
    sched.add_job(deleteFileInFolder, 'interval', minutes=1)
    sched.start()

