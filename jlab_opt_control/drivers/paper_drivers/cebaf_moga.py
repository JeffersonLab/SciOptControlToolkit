import numpy as np
from moga import MOGA
import time
import os
import argparse
import score.envs as envs
from pygmo import problem, population, unconstrain
import pickle

def run(env_id, n_iterations, result_loc, pop_size, index, save_interval=1000):
    # Run 16 trials on same machine
    for index in range(102, 104):
        logdir = "index_"+str(index)+"_NSGA_II_pop_"+str(pop_size)+"_Linac_"+str(env_id)+"nsteps_"+str(n_iterations)
        result_dir = os.path.join(result_loc, logdir)
        os.makedirs(result_dir, exist_ok=True)
        
        env = envs.make(env_id)
        
        
        heat_max = env.linac.max_allowed_heat
        trip_max = env.linac.max_allowed_trip
        ref = [heat_max, trip_max]
        
        lem_prob = MOGA(c_dim=3, c_ineq_dim=3, env=env, heat_max=heat_max, trip_max=trip_max)
        prob = problem(lem_prob)
        
        print('orignal problem:')
        print(prob)
        
        prob_dth = unconstrain(prob, method='kuri')  #'death penalty','kuri', 'weighted', 'ignore_c', 'ignore_o'   #????????????
        print(prob_dth)
        
        # Create original population
        pop = population(prob_dth)
        dim = prob.get_nx()
        x = np.empty(dim)
        for _ in range(pop_size):
            lem_prob.create_pop_w_constr(x)
            pop.push_back(x)
        print ("Initial pop generated!")
        
        final_results = {"heat":[], "trip":[], "alpha":[], "t_elapsed":[]}
        
        for N in range(1, n_iterations+1):
    
            start_time = time.time()
            pop = lem_prob.opt(pop, [1])
            time_taken = time.time() - start_time
    
            f = pop.get_f()
            results = np.array(f)
            final_results["heat"].append(results[:, 0])
            final_results["trip"].append(results[:, 1])
            final_results["alpha"].append(np.nan)
            final_results["t_elapsed"].append(time_taken)
        
            if N % save_interval == 0:
                with open(os.path.join(result_dir, env_id+'_nsga_II_results_'+str(N).zfill(6)+'.pkl'), "wb") as f:
                    pickle.dump(results, f)
                print("Results saved at iteration: ", N)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="registered env id", type=str, default="SCORE-MO-CEBAF-8D-VEC-v0")
    parser.add_argument("--n_iterations", help="number of training iterations", type=int, default=50000)
    parser.add_argument("--logdir", help="Location of directory where results shoule be saved", type=str, default="./paper_results/moga_8d/")
    parser.add_argument("--pop_size", help="Population size per generation", type=int, default=512)
    parser.add_argument("--index", help="index for result dir", type=int, default=0)

    
    # Get input arguments and overwrite the configuration
    args = parser.parse_args()
    
    env_id = getattr(args, "env")
    n_iterations = getattr(args, "n_iterations")
    logdir = getattr(args, "logdir")
    pop_size = getattr(args, "pop_size")
    index = getattr(args, "index")
    
    run(env_id, n_iterations, logdir, pop_size, index)
            
