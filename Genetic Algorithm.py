import numpy as np


TARGET = 'Cui Jie is the best person of the world'
POP_SIZE = 100
DNA_SIZE = len(TARGET)
CROSS_RATE = 0.5
MUTA_RATE = 0.04

TARGET_ASCII = np.fromstring(TARGET, dtype = np.uint8)
ASCII_BOUND = (32,126)

N_GENERATION = 1000000

class GA(object):
    def __init__(self, DNA_size, DNA_bound, pop_size, cross_rate, muta_rate):
        self.DNA_size = DNA_size
        self.pop_size = pop_size
        self.DNA_bound = (DNA_bound[0],DNA_bound[1]+1)
        self.cross_rate = cross_rate
        self.muta_rate = muta_rate



        self.pop = np.random.randint(*DNA_bound, size=(pop_size, DNA_size)).astype(np.int8)  # int8 for convert to ASCII

    def TranslationDNA(self,DNA):
        return DNA.tostring().decode('ascii')

    def Get_fittness(self):
        match_count = (self.pop == TARGET_ASCII).sum(axis = 1)
        return match_count


    def Selection(self,match_count):
        p_count = match_count + 1e-4
        p_sum = sum(p_count)
        idx = np.random.choice(np.arange(self.pop_size), size = self.pop_size,p=p_count/p_sum)
        return self.pop[idx]

    def Crossover(self, parent):
        if np.random.rand() < self.cross_rate:
            # i_ = np.random.randint(0, self.pop_size, size=1)                        # select another individual from pop
            i_ = np.argmax(self.Get_fittness())                                        #select the best_DNA
            cross_points = np.random.randint(0, 2, len(parent)).astype(np.bool)   # choose crossover points
            parent[cross_points] = self.pop[i_, cross_points]                            # mating and produce one child
        return parent


    def Mutation(self,DNA):
        muta_point = np.random.rand(len(DNA)) < self.muta_rate
        DNA[muta_point] = np.random.randint(self.DNA_bound[0],self.DNA_bound[1],size = self.DNA_size)[muta_point]
        return DNA

    def Evolute(self):
        fitness = self.Get_fittness()
        self.pop = self.Selection(fitness)
        for index_ in range(self.pop_size):
            next_gen = self.Crossover(self.pop[index_])
            next_gen = self.Mutation(next_gen)
            self.pop[index_][:] = next_gen




if __name__ == '__main__':
    ga = GA(DNA_size = DNA_SIZE, DNA_bound = ASCII_BOUND,pop_size = POP_SIZE, cross_rate = CROSS_RATE,muta_rate= MUTA_RATE)
    for i_ in range(1,N_GENERATION+1):

        ga.Evolute()
        fitness = ga.Get_fittness()
        best_DNA = ga.pop[np.argmax(fitness)]
        best_result = ga.TranslationDNA(best_DNA)
        print(i_,"generation:",best_result)
        if best_result == TARGET:
            break;







