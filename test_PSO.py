# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

MAX_ITER = 20000



def fit_function(x):
    f = (x[0]-10)**2 + 5*(x[1]-12)**2 + x[2]**4 + 3*(x[3]-11)**2 + 10*x[4]**6 + 7*x[5]**2 \
        + x[6]**4 - 4*x[5]*x[6] - 10*x[5] - 8*x[6]
    return f


def constraints(x, old_x=None):
    if old_x is None:
        if (np.max(x) > 10) or (np.min(x) < -10):
            x = 20*np.random.rand(len(x))-10
        g1 = 127 - 2*x[0]**2 - 3*x[1]**4 - x[2] - 4*x[3]**2 -5*x[4]
        g2 = 282 - 7*x[0] - 3*x[1] - 10*x[2]**2 - x[3] + x[4]
        g3 = 196 - 23*x[0] - x[1]**2 - 6*x[5]**2 + 8*x[6]
        g4 = -4*x[0]**2 - x[1]**2 + 3*x[0]*x[1] - 2*x[2]**2 - 5*x[5] + 11*x[6]

        while(g1 < 0 or g2 < 0 or g3 < 0 or g4 < 0):
            x = 20 * np.random.rand(len(x)) - 10
            g1 = 127 - 2*x[0]**2 - 3*x[1]**4 - x[2] - 4*x[3]**2 -5*x[4]
            g2 = 282 - 7*x[0] - 3*x[1] - 10*x[2]**2 - x[3] + x[4]
            g3 = 196 - 23*x[0] - x[1]**2 - 6*x[5]**2 + 8*x[6]
            g4 = -4*x[0]**2 - x[1]**2 + 3*x[0]*x[1] - 2*x[2]**2 - 5*x[5] + 11*x[6]
        return x
    else:
        if (np.max(x) > 10) or (np.min(x) < -10):
            return old_x
        g1 = 127 - 2*x[0]**2 - 3*x[1]**4 - x[2] - 4*x[3]**2 -5*x[4]
        g2 = 282 - 7*x[0] - 3*x[1] - 10*x[2]**2 - x[3] + x[4]
        g3 = 196 - 23*x[0] - x[1]**2 - 6*x[5]**2 + 8*x[6]
        g4 = -4*x[0]**2 - x[1]**2 + 3*x[0]*x[1] - 2*x[2]**2 - 5*x[5] + 11*x[6]
        if g1 < 0 or g2 < 0 or g3 < 0 or g4 < 0:
            return old_x
        return x



# -------------------Setting PSO Parameters-------------------------
class PSO():
    def __init__(self, p_num, dim, fit_function):
        self.omiga = 0.5
        self.alpha1 = 0.2
        self.alpha2 = 0.2
        self.p_num = p_num
        self.dim = dim
        self.max_iter = MAX_ITER
        self.X = 20 * np.random.rand(self.p_num, self.dim) - 10
        self.V = 20 * np.random.rand(self.p_num, self.dim) - 10
        self.pbest = np.array(self.X)
        self.p_fit = np.zeros(self.p_num)
        for i in range(self.p_num):
            self.X[i] = constraints(self.X[i])
            # print(self.p_num[i])
            self.p_fit[i] = fit_function(self.X[i])
        self.gbest = self.X[np.argmin(self.p_fit)]
        self.fit = np.min(self.p_fit)

    # ----------------------update the location----------------------------
    def iterator(self):
        fitness = []

        for t in range(self.max_iter):
            for i in range(self.p_num):
                self.V[i] = self.omiga * self.V[i] + self.alpha1 * (self.pbest[i] - self.X[i]) + \
                            self.alpha2 * (self.gbest - self.X[i])
                new_x = self.X[i] + self.V[i]
                self.X[i] = constraints(new_x, self.X[i])
                new_fit = fit_function(self.X[i])
                # update local best
                if new_fit < self.p_fit[i]:
                    self.p_fit[i] = new_fit
                    self.pbest[i] = self.X[i]

            # update global best
            self.gbest = self.X[np.argmin(self.p_fit)]
            self.fit = np.min(self.p_fit)
            fitness.append(self.fit)
            print("Iter: ", t, " Cost: ", self.fit,"Para: ", self.omiga,self.alpha1,self.alpha2)
        return fitness

my_pso = PSO(p_num=200, dim=7, fit_function = fit_function)

fitness = my_pso.iterator()
# -------------------figure--------------------
plt.figure(1)
plt.title("Find minimum of function")
plt.xlabel("iterators", size=14)
plt.ylabel("fitness", size=14)
t = np.array([t for t in range(MAX_ITER)])
fitness = np.array(fitness)
plt.plot(t, fitness, color='b', linewidth=3)
plt.show()