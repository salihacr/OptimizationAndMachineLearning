import numpy as np


def greiwank(x):
    return np.sum(np.power(x, 2))/4000 + 1 - np.product(np.cos(np.multiply(x, np.power(np.arange(1,len(x)+1), -0.5))))

def rastrigin(x):
    return np.sum(np.power(x, 2) - 10*np.cos(2*np.pi*x) + 10)

def rosenbrock(x):
    x2 = np.power(x, 2)
    v = np.sum(np.power(x2[:-1] - x[1:], 2)) + (x2[-1]*x[0])**2
    return 100*v + np.sum(np.power(1-x, 2))

def ackley(x):
    return 20+np.e-20*np.exp(-0.2*np.sqrt(np.sum(np.power(x, 2))/len(x)))-np.exp(np.sum(np.cos(2*np.pi*x))/len(x))

def schwefel(x):
    return len(x)*4128.9829-np.sum(x*np.sin(np.sqrt(np.abs(x))))

def sphere(x):
    return sum(np.power(x,2))