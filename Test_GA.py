import numpy as np


POP_SIZE = 300
DNA_SIZE = 7
CROSS_RATE = 0.6
MUTA_RATE = 0.04


N_GENERATION = 1000


def Get_fittness(x):
    f = (x[0] - 10) ** 2 + 5 * (x[1] - 12) ** 2 + x[2] ** 4 + 3 * (x[3] - 11) ** 2 + 10 * x[4] ** 6 + 7 * x[5] ** 2 \
        + x[6] ** 4 - 4 * x[5] * x[6] - 10 * x[5] - 8 * x[6]
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


class GA(object):
    def __init__(self, DNA_size, pop_size, cross_rate, muta_rate, Get_fittness):
        self.DNA_size = DNA_size
        self.pop_size = pop_size

        self.cross_rate = cross_rate
        self.muta_rate = muta_rate
        self.fittness = Get_fittness
        self.pop = 20*np.random.rand(self.pop_size, self.DNA_size)-10
        for i in range(self.pop_size):
            self.pop[i] = constraints(self.pop[i])

    # --------------------Update the iteration-----------------------

    def Get_fittness(self):
        match_count =[]
        for i in self.pop:
            match_count.append(self.fittness(i))
        return match_count

    def Selection(self,match_count):
        match_count = np.array(match_count)
        match_count = match_count/np.max(abs(match_count))

        p_count = np.exp(-match_count)
        p_sum = sum(p_count)
        idx = np.random.choice(np.arange(self.pop_size), size = self.pop_size,p=(p_count/p_sum))
        return self.pop[idx]

    def Crossover(self, parent):

        old_DNA = np.array(parent)
        new_DNA = parent[:]
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)                        # select another individual from pop
            # i_ = np.argmin(self.Get_fittness())                   #select the best_DNA
            cross_points = np.random.randint(0, 2, len(parent)).astype(np.bool)   # choose crossover points
            new_DNA[cross_points] = self.pop[i_, cross_points]
            new_DNA = constraints(new_DNA,old_DNA)

            # mating and produce one child
        return new_DNA

    def Mutation(self,DNA):
        old_DNA = np.array(DNA)
        muta_point = np.random.rand(len(DNA)) < self.muta_rate
        DNA[muta_point] = (20*np.random.rand(self.DNA_size)-10)[muta_point]
        DNA = constraints(DNA,old_DNA)
        return DNA

    def Evolute(self):
        fitness = self.Get_fittness()
        self.pop = self.Selection(fitness)
        for index_ in range(self.pop_size):
            next_gen = self.Crossover(self.pop[index_])
            next_gen = self.Mutation(next_gen)
            self.pop[index_][:] = next_gen




if __name__ == '__main__':
    ga = GA(DNA_size = DNA_SIZE,pop_size = POP_SIZE, cross_rate = CROSS_RATE,muta_rate= MUTA_RATE,Get_fittness=Get_fittness)
    for i_ in range(1,N_GENERATION+1):

        ga.Evolute()
        fitness = ga.Get_fittness()
        best_DNA = ga.pop[np.argmin(fitness)]
        best_result = Get_fittness(best_DNA)
        print(i_,"generation:",best_result)






