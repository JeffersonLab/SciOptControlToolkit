from pygmo import algorithm, nsga2
import numpy as np
import random


class MOGA:
    
    def __init__(self, c_dim, c_ineq_dim, env, heat_max, trip_max):

        self.env = env

        # Set bounds
        self.range_high = self.env.max_grads
        self.range_low = self.env.min_grads
        
        self.trip_max = trip_max
        self.c_dim = c_dim
        self.c_ineq_dim = c_ineq_dim
        self.heat_max = heat_max
        self.A = -10.268 #-10.26813067 
        
    def fitness(self, xx):
        action = self.env.normalize_action(np.array(xx))
        _, _, _, _, info = self.env.step(action)
        
        diff_energy = np.fabs(info['energy'] - self.env.target_energy)
        
        obj = [info['heat'], info['trip']]
        
        ci = []
        if self.c_dim == 1:
            ci = [diff_energy - self.env.delta_energy]
        elif self.c_dim == 2:
            ci = [diff_energy - self.env.delta_energy, info['trip'] - self.trip_max]
        elif self.c_dim == 3:
            ci = [diff_energy - self.env.delta_energy, info['trip'] - self.trip_max, info['heat'] - self.heat_max]
        else:
            print("Warning: Constraint number should be ONE, TWO, or THREE!")
        return obj + ci

    def get_nobj(self):
        return 2
    
    def get_bounds(self):
        return (self.range_low, self.range_high)
    
    def get_nic(self):
        return self.c_ineq_dim
    
    def get_nec(self):
        return self.c_dim - self.c_ineq_dim
    
    def get_nc(self):
        return self.c_dim

    def __len__(self):
        return self.Q.size

    # Prepare the population
    def create_pop(self, x):
        for i, xi in enumerate(x):
            x[i] = (self.range_high[i] - self.range_low[i]) * random.random() + self.range_low[i]
        
        _, _, terminate, _, info = self.env.step(self.env.normalize_action(np.array(x)))
        
        while terminate:
            current_energy = info['energy']
            rate = 1 + (self.env.target_energy - current_energy) / current_energy
            if (rate > 1):
                for i, xi in enumerate(x):
                    x[i] = x[i] * rate
                    if (x[i] > self.range_high[i]):
                        x[i] = self.range_high[i] * (1 - 1e-6)
            if (rate < 1):
                for i, xi in enumerate(x):
                    x[i] = x[i] * rate
                    if (x[i] < self.range_low[i]):
                        x[i] = self.range_low[i] * (1 + 1e-6)
            _, _, terminate, _, info = self.env.step(self.env.normalize_action(np.array(x)))
            current_energy = info['energy']

    def create_pop_w_constr(self, x): 
        self.create_pop(x)
        
        _, _, _, _, info = self.env.step(self.env.normalize_action(np.array(x)))

        if self.c_dim == 2:
            while info['trip'] > self.trip_max:
                self.create_pop(x)
                _, _, _, _, info = self.env.step(self.env.normalize_action(np.array(x)))
        elif self.c_dim == 3:
            while info['trip'] > self.trip_max and info['heat'] > self.heat_max:
                self.create_pop(x)
                _, _, _, _, info = self.env.step(self.env.normalize_action(np.array(x)))
                
    # Define the algorithm
    def opt(self, pop, ngen):

        if (isinstance(ngen, np.ndarray)):
            ngen = ngen.tolist()

        n_gen = [0] + ngen    
        t = np.zeros(len(n_gen))

        for i in range(len(n_gen)-1):
            algo = algorithm(nsga2(n_gen[i+1]-n_gen[i], m=0.01, cr = 0.95))
            pop = algo.evolve(pop)

        return pop