import os
import glob
from io import BytesIO
from datetime import datetime

import random
import string
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

# Compare and Graph the Routes of Algorithms.
def plt_compare_routes(x_axis, y_axis, cities, best_routes, colors, labels):
    fig, ax = plt.subplots(1, figsize=(12, 8), dpi=200)
    fig.suptitle('Compare Optimization Algorithms for TSP Problem')
    plt.xlabel('X AXIS')
    plt.ylabel('Y AXIS')

    # city points
    ax.scatter(x_axis, y_axis, c='red', s=150)
    # and city names
    for i in range(len(cities)):
        ax.annotate(cities[i], xy = get_XY_location(
            x_axis, y_axis, i), c='black')

    for i in range(0, len(best_routes)):
        # algorithm path
        algorithm_path = np.append(best_routes[i], best_routes[i])

        # adding paths to plot, path plot
        plt.plot(x_axis[algorithm_path], y_axis[algorithm_path], lw = 2.5,
                c = colors[i], label = labels[i])

        algorithm_path = 0
    plt.legend()
    return fig

# Compare and Graph the Cost of Algorithms.
def plt_compare_costs(cost_values, colors, labels):
    fig, ax = plt.subplots(1, dpi=200)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    fig.suptitle('Compare Optimization Algorithms Costs')
    for i in range(0, len(cost_values)):
        ax.plot(cost_values[i], 'r--', c=colors[i], label=labels[i])
    plt.legend()
    return fig

# This function returns the random string to change the name of the picture.
def get_random_string(length):
    letters = string.ascii_letters
    value = random.randint(100, 999)
    result_str = ''.join(random.choice(letters) for i in range(length))
    result_str += str(value)
    result_str += '_'
    #print('Random string of length', length, 'is:', result_str)
    return result_str

# This function returns the current time to change the name of the picture.
def get_current_time():
    return str(datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f_')[:-3])

#This function combine random string and current time
def combine_names():
    return str(get_current_time() + get_random_string(10))

#This function save image to given folder.
# using >  plt_img1_path = save_figures_to_upload(figure,your_img_name)
def save_figures_to_upload(plot_fig, img_name):
    rand_str = get_random_string(10)
    current_time = get_current_time()
    temp_img_name = rand_str + current_time + img_name

    base_path = 'static/uploads/'
    img_path = base_path + temp_img_name
    
    plt.savefig(img_path)
    #print('Image Path => ', img_path)

    return img_path

def format_for_genetic(longitudes_x, latitudes_y):
    distance_list = zip(longitudes_x, latitudes_y)
    return list(distance_list)

def delete_files_in_folder():
    dir = 'static/uploads'
    filelist = glob.glob(os.path.join(dir, '*'))
    if len(filelist) != 0:
        for f in filelist:
            os.remove(f)

def delete_files_in_folder_by_dir(dir=''):
    temp_dir = dir
    filelist = glob.glob(os.path.join(temp_dir, '*'))
    if len(filelist) != 0:
        for f in filelist:
            os.remove(f)

#This function runs the given job at scheduled times.
def run_schedule():
    sched = BackgroundScheduler(daemon=True)
    sched.add_job(delete_files_in_folder,
                  'interval', minutes=30)
    sched.start()