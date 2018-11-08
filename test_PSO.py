# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import decimal
from functools import reduce

MAX_ITER = 300

dim1 = 7
dim2 = 5
dim3 = 10

# 300pop  0.5 0.2 0.2
def fit_function1(x):
    f = (x[0]-10)**2 + 5*(x[1]-12)**2 + x[2]**4 + 3*(x[3]-11)**2 + 10*x[4]**6 + 7*x[5]**2 \
        + x[6]**4 - 4*x[5]*x[6] - 10*x[5] - 8*x[6]
    return f
def constraints1(x, old_x=None):
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


def fit_function2(x):
    f = 5.3578547*x[2]**2 + 0.8356891*x[0]*x[4] + 37.293239*x[0] - 40729.141
    return f
def constraints2(x, old_x=None):
    if old_x is None:
        x[0] = 78 + np.random.rand()*(102 - 78)
        x[1] = 33 + np.random.rand()*(45 - 33)
        x[2] = 27 + np.random.rand()*(45 - 27)
        x[3] = 27 + np.random.rand()*(45 - 27)
        x[4] = 27 + np.random.rand()*(45 - 27)
        g1 = 85.334407 + 0.0056858*x[1]*x[4] + 0.0006262*x[0]*x[3] - 0.0022053*x[2]*x[4] - 92
        g2 = -85.334407 - 0.0056858*x[1]*x[4] - 0.0006262*x[0]*x[3] - 0.0022053*x[2]*x[4]
        g3 = 80.51249 + 0.0071317*x[1]*x[4] + 0.0029955*x[0]*x[1] + 0.0021813*x[2]**2 - 110
        g4 = -80.51249 - 0.0071317*x[1]*x[4] - 0.0029955*x[0]*x[1] - 0.0021813*x[2]**2 + 90
        g5 = 9.300961 + 0.0047026*x[2]*x[4] + 0.0012547*x[0]*x[2] + 0.0019085*x[2]*x[3] - 25
        g6 = -9.300961 - 0.0047026 * x[2] * x[4] - 0.0012547 * x[0] * x[2] - 0.0019085 * x[2] * x[3] + 20

        while(g1>0 or g2>0 or g3>0 or g4>0 or g5> 0 or g6>0):
            x[0] = 78 + np.random.rand() * (102 - 78)
            x[1] = 33 + np.random.rand() * (45 - 33)
            x[2] = 27 + np.random.rand() * (45 - 27)
            x[3] = 27 + np.random.rand() * (45 - 27)
            x[4] = 27 + np.random.rand() * (45 - 27)
            g1 = 85.334407 + 0.0056858 * x[1] * x[4] + 0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4] - 92
            g2 = -85.334407 - 0.0056858 * x[1] * x[4] - 0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4]
            g3 = 80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2] ** 2 - 110
            g4 = -80.51249 - 0.0071317 * x[1] * x[4] - 0.0029955 * x[0] * x[1] - 0.0021813 * x[2] ** 2 + 90
            g5 = 9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3] - 25
            g6 = -9.300961 - 0.0047026 * x[2] * x[4] - 0.0012547 * x[0] * x[2] - 0.0019085 * x[2] * x[3] + 20
        return x
    else:
        bool_x0 = (78 > x[0]) or (x[0] > 102)
        bool_x1 = (33 > x[1]) or (x[1] > 45)
        bool_x2 = (27 > x[2]) or (x[2] > 45)
        bool_x3 = (27 > x[3]) or (x[3] > 45)
        bool_x4 = (27 > x[4]) or (x[4] > 45)
        if (bool_x0 or bool_x1 or bool_x2 or bool_x3 or bool_x4):
            return old_x
        g1 = 85.334407 + 0.0056858 * x[1] * x[4] + 0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4] - 92
        g2 = -85.334407 - 0.0056858 * x[1] * x[4] - 0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4]
        g3 = 80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2] ** 2 - 110
        g4 = -80.51249 - 0.0071317 * x[1] * x[4] - 0.0029955 * x[0] * x[1] - 0.0021813 * x[2] ** 2 + 90
        g5 = 9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3] - 25
        g6 = -9.300961 - 0.0047026 * x[2] * x[4] - 0.0012547 * x[0] * x[2] - 0.0019085 * x[2] * x[3] + 20
        if g1>0 or g2>0 or g3> 0 or g4>0 or g5>0 or g6>0:
            return old_x
        return x



# def fit_function3(x):
#     f = 0
#     for i in range(len(x)):
#         f += - (np.power())
#     return f
def constraints3(x, old_x=None):
    if old_x is None:
        if (np.max(x) > 10) or (np.min(x) < -10):
            x = 20*np.random.rand(len(x))-10
        g1 = 127 - 2*x[0]**2 - 3*x[1]**4 - x[2] - 4*x[3]**2 -5*x[4]
        g2 = 282 - 7*x[0] - 3*x[1] - 10*x[2]**2 - x[3] + x[4]
        g3 = 196 - 23*x[0] - x[1]**2 - 6*x[5]**2 + 8*x[6]
        g4 = -4*x[0]**2 - x[1]**2 + 3*x[0]*x[1] - 2*x[2]**2 - 5*x[5] + 11*x[6]

        while(g1 > 0 or g2 > 0 or g3 > 0 or g4 > 0):
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
        self.omiga = 0.0
        self.alpha1 = 0.4
        self.alpha2 = 0.6
        self.p_num = p_num
        self.dim = dim
        self.max_iter = MAX_ITER
        self.X = np.zeros((self.p_num,self.dim))
        self.V = np.zeros((self.p_num,self.dim))
        self.p_fit = np.zeros(self.p_num)
        for i in range(self.p_num):
            self.X[i] = constraints2(self.X[i])
            # print(self.p_num[i])
            self.p_fit[i] = fit_function(self.X[i])
        self.pbest = np.array(self.X)
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
                self.X[i] = constraints2(new_x, self.X[i])
                new_fit = fit_function2(self.X[i])
                # update local best
                self.p_fit[i] = new_fit
                if new_fit < self.p_fit[i]:
                    self.pbest[i] = self.X[i]

            # update global best
            self.gbest = self.X[np.argmin(self.p_fit)]
            self.fit = np.min(self.p_fit)
            fitness.append(self.fit)
            print("Iter: ", t, " Cost: ", self.fit,"Para: ", self.omiga,self.alpha1,self.alpha2)
        return fitness


if __name__ == '__main__':
    # fitness_list = []
    # t_range = 10
    # for t in range(t_range):
    #     my_pso = PSO(p_num=100, dim=dim1, fit_function = fit_function1)
    #
    #     fitness = my_pso.iterator()
    #     fitness_list.append(fitness[-1])
    # print("fitness each time:",fitness_list)
    # print("mean_fitness:",np.mean(fitness_list),"Best_fitness:",np.min(fitness_list))
    # # -------------------figure--------------------
    # plt.figure(1)
    # plt.title("Find minimum of function")
    # plt.xlabel("iterators", size=14)
    # plt.ylabel("fitness", size=14)
    # # t = np.array([t for t in range(MAX_ITER)])
    # fitness = np.array(fitness)
    # # plt.plot(t, fitness, color='b', linewidth=3)
    # for x, y in zip(range(t_range), fitness_list):
    #     y2 = decimal.Decimal(y).quantize(decimal.Decimal('0.0000'))
    #     plt.text(x, y+0.01, str(y2), ha='center', va='bottom')
    # plt.plot(range(t_range),fitness_list,color = 'r')
    # plt.savefig("PSO.pdf")
    # plt.show()

    # ------------------fit_function2----------------------
    fitness_list = []
    t_range = 1
    for t in range(t_range):
        my_pso = PSO(p_num=200, dim=dim2, fit_function = fit_function2)
        fitness = my_pso.iterator()
        fitness_list.append(fitness[-1])
    print("fitness each time:",fitness_list)
    print("mean_fitness:",np.mean(fitness_list),"Best_fitness:",np.min(fitness_list))
    # -------------------figure--------------------
    plt.figure(1)
    plt.title("Find minimum of function")
    plt.xlabel("iterators", size=14)
    plt.ylabel("fitness", size=14)
    # t = np.array([t for t in range(MAX_ITER)])
    fitness = np.array(fitness)
    # plt.plot(t, fitness, color='b', linewidth=3)
    for x, y in zip(range(t_range), fitness_list):
        y2 = decimal.Decimal(y).quantize(decimal.Decimal('0.0000'))
        plt.text(x, y+0.01, str(y2), ha='center', va='bottom')
    plt.plot(range(t_range),fitness_list,color = 'r')
    plt.savefig("PSO.pdf")
    plt.show()