from algorithms.optimization.obj_functions import*
from algorithms.optimization.abc import Bee, ABC, plot_results
from algorithms.optimization.sa import SimulatedAnnealing
from algorithms.optimization.pso import PSO
from algorithms.optimization.dea import DifferentialEvolution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import os, random, math, time, timeit
from io import BytesIO

from helpers import helper
from algorithms.tsp.aco_tsp_solve import ACO_TSP_SOLVE
from algorithms.tsp.ga_tsp_solve import GA_TSP_SOLVE
from algorithms.tsp.abc_tsp_solve import ABC_TSP_SOLVE

from flask import Flask, render_template, request, make_response, redirect, abort, flash, url_for, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config['SECRET_KEY'] = '^%huYtFd90;90jjj'
app.config['UPLOADED_DATAS'] = 'static/uploads'

# Cache Blocker
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age = 0'
    return r

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error = "Page Not Found", error_code = 404), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error = "Internal Server Error", error_code = 500), 500

helper.run_schedule() #Delete Img

@app.route('/')
def index():
    return render_template("index.html")
@app.route('/optimization')
def optimization():
    return render_template("optimization.html")
@app.route('/machine_learning')
def machine_learning():
    return render_template("machine-learning.html")
@app.route('/travellingsalesmanproblem')
def tsp():
    return render_template("tsp.html")

@app.route('/salih')
def salih():
    return render_template("salih.html")

@app.route('/optimize', methods = ['POST'])
def optimize():
    try:
        opt_cost_list = []
        opt_color_list = []
        opt_label_list = []

        # Ortaklar: Bounds, iteration, objFunc, problem size
        bound = request.form.get("bound", False)
        boundConvert = float(bound)
        bounds = [-boundConvert, boundConvert]
        print("Bounds Aralık : ", bounds)

        iter = request.form.get("iteration", False)
        iteration = int(iter)
        print("İt Sayısı : ", iteration)
        
        selectedAlg = request.form.getlist('alg_checkbox')
        print("Seçilen Algoritmalar:  ", selectedAlg)

        if len(selectedAlg) < 2 :
            print("Boş Geldi")
            return jsonify({'error': 'Karşılaştırmak için en az 2 algoritma seçiniz !'})
        else:
            obj_func = request.form.get("obj_function", False)
            print("Seçilen Test Fonksiyonu:  ", obj_func)

            if obj_func ==  'sphere':
                obj_function = sphere
            elif obj_func ==  'rastrigin':
                obj_function = rastrigin
            elif obj_func ==  'rosenbrock':
                obj_function = rosenbrock
            elif obj_func ==  'griewank':
                obj_function = griewank
            else:
                obj_function = sphere

            print("Seçilen Fonksiyon: ", obj_function)
            prob_size = request.form.get("problem_size", False)
            problem_size = int(prob_size)
            print("Problem Boyutu", problem_size)

            if bound and iter and prob_size and obj_function:
                # ---------------SA Alg---------------
                dlt = request.form.get("sa_delta", False)
                delta = float(dlt)
                print("SA - Delta: ", delta)

                sa = SimulatedAnnealing(obj_function, problem_size = problem_size, bounds = bounds,
                                        iteration = iteration, temperature = 10000, cooling_coefficient = 0.99, delta = delta)
                sa_cost_values, sa_best_cost, sa_best_solve = sa.run_optimize()

                opt_cost_list.append(sa_cost_values)
                opt_color_list.append('blue')
                opt_label_list.append('SA-SimulatedAnnealing')

                # ---------------Difer Alg---------------
                mr = request.form.get("dea_mutation_rate", False)
                mutation_rate = float(mr)
                print("DE - Mutasyon Oranı:", mutation_rate)

                cr = request.form.get("dea_crossRate", False)
                cross_rate = float(cr)
                print("DE - Çaprazlama Oranı:", mutation_rate)

                ps = request.form.get("dea_population_size", False)
                population_size = int(ps)
                print("DE - Dif Alg Pop Boyutu", population_size)

                de = DifferentialEvolution(obj_function, bounds = bounds, iteration = iteration, population_size = population_size,
                                        problem_size = problem_size, mutation_rate = mutation_rate, cr = cross_rate)
                de_cost_values, de_best_cost, de_best_solution = de.run_optimize()

                # 'red','green','blue','orange','black','purple'
                opt_cost_list.append(de_cost_values)
                opt_color_list.append('red')
                opt_label_list.append('DE-Differential Evolution')

                # SA - DE 2'li karşılaştırma
                plt_compare_fig = helper.plt_compare_costs(
                    cost_values = opt_cost_list, colors = opt_color_list, labels = opt_label_list)
                plt_compare_fig_path = helper.save_figures_to_upload(
                    plot_fig = plt_compare_fig, img_name = "plt_compare_cost.png")

                # ---------------Pso Alg---------------
                part_size = request.form.get("pso_particle_size", False)
                partical_size = int(part_size)
                print("PSO - Sürü Boyutu:", partical_size)

                w = request.form.get("pso_weight", False)
                weights = float(w)
                print("PSO - Pso Ağırlık:", weights)

                pso = PSO(obj_function, bounds = bounds, iteration = iteration,
                        problem_size = problem_size, particle_size = partical_size, w = weights)
                pso_cost_values, pso_best_value = pso.run_optimize()

                opt_cost_list.append(pso_cost_values)
                opt_color_list.append('green')
                opt_label_list.append('PSO-Particle Swarm')

                # ---------------Abc Alg---------------
                bee_num = request.form.get("abc_populationSize", False)
                bee_size = int(bee_num)
                print("ABC - Arı Sayısı:", bee_size)

                lmt = request.form.get("abc_limit", False)
                limit = float(lmt)
                print("ABC - Arı Limit:", limit)

                abc_best, abc_cost_values = ABC(
                    obj_function, bounds = bounds, iteration = iteration, problem_size = problem_size, bee_size = bee_size, limit = limit)
                print("Best Value: ", abc_best.f)
                print("cost_values:", abc_cost_values)

                opt_cost_list.append(abc_cost_values)
                opt_color_list.append('orange')
                opt_label_list.append('ABC-Artifical Bee Colony')

                # --------------------------BİTTİ--------------------------
                # Save Dife Alg
                plt_de_costs_fig = de.plot_results(de_cost_values)
                de_cost_fig_path = helper.save_figures_to_upload(
                    plot_fig = plt_de_costs_fig, img_name = "plt_de_cost.png")

                # Save Abc
                plt_abc_costs_fig = plot_results(abc_cost_values)
                abc_cost_fig_path = helper.save_figures_to_upload(
                    plot_fig = plt_abc_costs_fig, img_name = "plt_abc_cost.png")

                # Save Pso
                plt_pso_costs_fig = pso.plot_results(pso_cost_values)
                pso_cost_fig_path = helper.save_figures_to_upload(
                    plot_fig = plt_pso_costs_fig, img_name = "plt_pso_cost.png")

                # Save Sa
                plt_sa_costs_fig = sa.plot_results(sa_cost_values)
                sa_cost_fig_path = helper.save_figures_to_upload(
                    plot_fig = plt_sa_costs_fig, img_name = "plt_sa_cost.png")

                # ALL - 4'lü karşılaştırma.
                plt_all_compare_fig = helper.plt_compare_costs(
                    cost_values = opt_cost_list, colors = opt_color_list, labels = opt_label_list)
                plt_all_compare_fig_path = helper.save_figures_to_upload(
                    plot_fig = plt_compare_fig, img_name = "plt_all_compare_cost.png")

                user_compare_algo = []
                user_compare_color = []
                user_compare_label = []

                for i in range(len(selectedAlg)):
                    if selectedAlg[i] == 'abc':
                        user_compare_algo.append(abc_cost_values)
                        user_compare_color.append('orange')
                        user_compare_label.append('ABC-Artifical Bee Colony')
                    
                    if selectedAlg[i] == 'sa':
                        user_compare_algo.append(sa_cost_values)
                        user_compare_color.append('blue')
                        user_compare_label.append('SA-SimulatedAnnealing')

                    if selectedAlg[i] == 'pso':
                        user_compare_algo.append(pso_cost_values)
                        user_compare_color.append("green")
                        user_compare_label.append('PSO-Particle Swarm')    

                    if selectedAlg[i] == 'dea':
                        user_compare_algo.append(de_cost_values)
                        user_compare_color.append('red')
                        user_compare_label.append('DE-Differential Evolution')       
    
                plt_compare_fig2 = helper.plt_compare_costs(
                    cost_values = user_compare_algo, colors = user_compare_color, labels = user_compare_label)
                plt_compare_fig_path2 = helper.save_figures_to_upload(
                    plot_fig = plt_compare_fig2, img_name = "plt_compare_cost.png")

                return jsonify({
                    'compare_costs_path2': plt_compare_fig_path2,
                    'all_compare_costs_path': plt_all_compare_fig_path,
                    'de_cost_path': de_cost_fig_path,
                    'abc_cost_path': abc_cost_fig_path,
                    'pso_cost_path': pso_cost_fig_path,
                    'sa_cost_path': sa_cost_fig_path})

            return jsonify({'error': 'Lütfen tüm verileri eksiksiz doldurun !'})
        return "done"
    except :
        return jsonify({'error': 'Beklenmedik bir hata meydana geldi. Lütfen tekrar deneyin. !'})

# ----------------------------TSP----------------------------
@app.route('/solvetsp', methods = ['POST'])
def solve_tsp():
    try:
        """
        file operation done
        """
        file = request.files['cityfile']
        print("dosya adı", file.filename)
        if file.filename:

            filename = helper.combine_names()
            filename +=  secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOADED_DATAS'], filename))

            # file name
            print(file.filename, type(file), file.filename.split('.')[0])

            # file extension -1 yapıyor
            print(file.filename, type(file), file.filename.split('.')[-1])

            file_extension = str(file.filename.split('.')[-1])
            
            if file_extension ==  "txt":
                df = pd.read_text("static/uploads/{}".format(filename))
            if file_extension ==  "csv":
                df = pd.read_csv("static/uploads/{}".format(filename))

        else:
            ready_map = request.form.get("map", False)
            if  ready_map == 'marmara':
                filename = 'marmara_mesafeler.csv'

            if ready_map == 'icanadolu':
                filename = 'icanadolu_mesafeler.csv'

            if ready_map == 'karadeniz':
                filename = 'karadeniz_mesafeler.csv'  

            if ready_map == 'doguanadolu':
                filename = 'dogu_anadolu_mesafeler.csv'
            
            if ready_map == 'guneydogu':
                filename = 'guneydogu_mesafeler.csv'
            
            if ready_map == 'akdeniz':
                filename = 'akdeniz_mesafeler.csv'
            
            if ready_map == 'ege':
                filename = 'ege_mesafeler.csv'
            
            if ready_map == 'anothercity':
                filename = 'anothercity_locations.csv'
            df = pd.read_csv("data/{}".format(filename))
            df.head()
        """
        end file operation
        """        
        # dataset parameters (city)
        cities = df.loc[:, 'city'].values
        cities_x_axis = df.loc[:, 'longitude'].values  # longitude is x axis
        cities_y_axis = df.loc[:, 'latitude'].values  # latitude is y axis

        # ant colony optimization paramaters
        ant_size = request.form.get('ant_size', False)
        pheromone_rho = request.form.get('pheromone_rho', False)
        alpha = request.form.get('alpha', False)
        beta = request.form.get('beta', False)

        # genetic algorithm parameters
        life_count = request.form.get('life_count', False)
        mutation_rate = request.form.get('mutation_rate', False)
        cross_rate = request.form.get('cross_rate', False)

        # artificial bee colony optimization paramaters
        number_of_bees = request.form.get('number_of_bees', False)
        number_of_worker_bees = request.form.get('number_of_worker_bees', False)
        limit = request.form.get('limit', False)

        # iteration
        iteration = request.form.get('iteration', False)

        # file da eklenecek
        if int(ant_size) < len(cities):
            return jsonify({'error': 'Karınca sayısı {} den küçük olamaz !'.format(len(cities))})
        elif int(life_count) < len(cities):
            return jsonify({'error': 'Genetik Algoritmadaki Birey sayısı {} den küçük olamaz !'.format(len(cities))})
        elif int(number_of_bees) < len(cities):
            return jsonify({'error': 'Arı sayısı {} den küçük olamaz !'.format(len(cities))})

        elif ant_size and pheromone_rho and alpha and beta and life_count and mutation_rate and cross_rate and number_of_bees and number_of_worker_bees and limit and iteration:

            cities_xy = helper.format_for_genetic(cities_x_axis, cities_y_axis)

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

            ga_best_route, ga_cost_values, ga_best_cost = ga_tsp.run(
                int(iteration))

            abc_tsp = ABC_TSP_SOLVE(x_axis = cities_x_axis, y_axis = cities_y_axis,
                                    number_of_bees = int(number_of_bees), number_of_worker_bees = int(number_of_worker_bees),
                                    limits_of_try = int(limit), iteration = int(iteration), cities = cities)

            abc_best_route, abc_cost_values, abc_best_cost = abc_tsp.start_optimize()

            print(abc_best_cost, aco_best_cost, ga_best_cost)

            # save ant colony route figure
            plt_antcolony_route_fig = aco_tsp.plot_cities(aco_best_route, aco_cost_values)
            antcolony_route_fig_path = helper.save_figures_to_upload(plot_fig = plt_antcolony_route_fig, img_name = "plt_antcolony_route.png")

            # save ant colony cost figure
            plt_antcolony_costs_fig = aco_tsp.plot_cost_iteration(aco_cost_values)
            antcolony_cost_fig_path = helper.save_figures_to_upload(plot_fig = plt_antcolony_costs_fig, img_name = "plt_antcolony_costs.png")

            # save genetic algorithm route figure
            plt_ga_route_fig = ga_tsp.plot_cities(cities_x_axis, cities_y_axis, ga_best_route, ga_cost_values)
            ga_route_fig_path = helper.save_figures_to_upload(plot_fig = plt_ga_route_fig, img_name = "plt_ga_route.png")

            # save genetic algorithm cost figure
            plt_ga_costs_fig = ga_tsp.plot_cost_iteration(ga_cost_values)
            ga_cost_fig_path = helper.save_figures_to_upload(plot_fig = plt_ga_costs_fig, img_name = "plt_ga_costs.png")

            # save artificial bee colony algorithm route figure
            plt_abc_route_fig = abc_tsp.plot_cities(abc_best_route, abc_cost_values)
            abc_route_fig_path = helper.save_figures_to_upload(plot_fig = plt_abc_route_fig, img_name = "plt_abc_route.png")

            # save artificial bee colony cost figure
            plt_abc_costs_fig = abc_tsp.plot_cost_iteration(abc_cost_values)
            abc_cost_fig_path = helper.save_figures_to_upload(plot_fig = plt_abc_costs_fig, img_name = "plt_abc_costs.png")
            
            route_list = [aco_best_route, ga_best_route, abc_best_route]
            cost_list = [aco_cost_values, ga_cost_values, abc_cost_values]
            color_list = ['green', 'blue', 'orange']
            label_list = ['Ant Colony', 'Genetic Algorithm', 'Bee Colony']
            
            # save compare routes figure
            plt_compare_routes_fig =helper.plt_compare_routes(x_axis = cities_x_axis, y_axis = cities_y_axis,
            cities= cities, best_routes = route_list, colors = color_list, labels = label_list)

            compare_route_fig_path = helper.save_figures_to_upload(plot_fig = plt_compare_routes_fig, img_name = "plt_compare_routes.png")

            # save compare cost figure
            plt_compare_costs_fig = helper.plt_compare_costs(cost_list, color_list, label_list)
            compare_cost_fig_path = helper.save_figures_to_upload(plot_fig = plt_compare_costs_fig, img_name = "plt_compare_costs.png")

            return jsonify({'compare_routes_fig_path': compare_route_fig_path,
                            'compare_costs_fig_path': compare_cost_fig_path,
                            'antcolony_route_path': antcolony_route_fig_path,
                            'antcolony_cost_path': antcolony_cost_fig_path,
                            'beecolony_route_path': abc_route_fig_path,
                            'beecolony_cost_path': abc_cost_fig_path,
                            'ga_route_path': ga_route_fig_path,
                            'ga_cost_path': ga_cost_fig_path})

        return jsonify({'error': 'Lütfen tüm verileri eksiksiz doldurun !'})
    except :
        return jsonify({'error': 'Beklenmedik bir hata meydana geldi. Lütfen tekrar deneyin. !'})


if __name__ ==  "__main__":
    app.run(debug = True)