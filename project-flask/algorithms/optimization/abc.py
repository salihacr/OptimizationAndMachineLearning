import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

class Bee:
    def __init__(self, x=[], f=None):
        self.x = x
        self.f = f

def ABC(objective_function, bounds = (-5.12, 5.12), iteration = 4, 
problem_size = 4, bee_size = 10, limit = 10):

    (xmin, xmax) = bounds

    # Scout bees initiate food source, random food positions
    X = np.random.uniform(low=xmin, high=xmax, size=(iteration, problem_size))
    employed = [Bee(x=X[i], f=objective_function(X[i])) for i in range(iteration)]
    onlooker = employed[:]

    C = np.zeros(iteration)
    for it in range(bee_size):

        for i in range(iteration):
            K = list(range(i-1))+list(range(i, iteration))
            k = K[np.random.randint(len(K))]

            phi = np.random.uniform(low=-1, high=1, size=problem_size)
            v = employed[i].x + np.multiply(phi, (employed[i].x - employed[k].x))
            v = np.minimum(np.maximum(v, xmin), xmax)

            fv = objective_function(v)
            
            if fv < employed[i].f:
                employed[i] = Bee(x=v, f=fv)
            else:
                C[i] += 1
        
        # Onlooker bees; place on the food sources in the memory
        # -> select the food sources
        fit = np.zeros(iteration)
        for i in range(iteration):

            if (employed[i].f >= 0):
                fit[i] = 1/(1+employed[i].f)
            else:
                fit[i] = 1+abs(employed[i].f)
        P = fit/sum(fit)
        
        for i in range(iteration):

            n = np.random.choice(range(len(P)), p=P/np.sum(P))
            
            K = list(range(n-1))+list(range(n, iteration))
            k = K[np.random.randint(len(K))]

            phi = np.random.uniform(low=-1, high=1, size=problem_size)
            v = employed[n].x + np.multiply(phi, (employed[n].x - employed[k].x))
            v = np.minimum(np.maximum(v, xmin), xmax)

            fv = objective_function(v)
            
            if fv < onlooker[n].f:
                onlooker[n] = Bee(x=v, f=fv)

        mask = C >= limit
        tot_exh = sum(mask)
        if tot_exh > 0:
            i = np.random.choice(range(iteration), p=mask/tot_exh)
            employed[i].x = np.random.uniform(low=xmin, high=xmax, size=problem_size)
            employed[i].f = objective_function(employed[i].x)
            C[i] = 0
    
    cost_values = []

    best_bees = Bee(f=float('inf'))

    for i in range(iteration):
        if employed[i].f < best_bees.f: best_bees = employed[i]
        if onlooker[i].f < best_bees.f: best_bees = onlooker[i]
        cost_values.append(best_bees.f)
    
    return best_bees, cost_values



def plot_results(cost_values):
    fig, ax = plt.subplots(1, dpi=200)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    ax.plot(cost_values, "r--", c="orange", label = 'Artificial Bee Colony')
    plt.legend()
    # plt.show()
    return fig