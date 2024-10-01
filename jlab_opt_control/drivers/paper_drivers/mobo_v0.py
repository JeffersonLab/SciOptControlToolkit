import platform
print("Platform: ", platform.processor())
# Third party imports
import numpy as np
import torch
import pandas as pd
import time
import argparse
import os
import pickle

# Internal imports
import score.envs as envs
from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import MOBOGenerator


def run(env_id, n_iterations, logdir, warmup_size, index):
    result_dir = "trial_"+str(index) #"index_"+str(index)+"_MOBO_Linac_"+str(env_id)+"nsteps_"+str(n_iterations)
    logdir = os.path.join(logdir, result_dir)
    os.makedirs(logdir, exist_ok=True)
    # Make environment
    env = envs.make(env_id)
    env.energy_penalty = False
    env.reset()
    
    def eval_cebaf(action_dict):
        
        if "-8D-" in env_id:
            action = np.array([action_dict["1L10-1"],
                                            action_dict["1L10-2"],
                                            action_dict["1L10-3"],
                                            action_dict["1L10-4"],
                                            action_dict["1L10-5"],
                                            action_dict["1L10-6"],
                                            action_dict["1L10-7"],
                                            action_dict["1L10-8"]]).flatten()
        elif "-N-" in env_id:
            action = np.array(list(action_dict.values())).flatten()

        inf_next_state, inf_reward, inf_terminate, inf_truncate, inf_info = env.step(action)     
        
        outdict = {
            "heat": inf_info['heat'],
            "trip": inf_info['trip'],
            "energy": inf_info['energy'],
            "heat1": inf_info['heat'],
            "trip1": inf_info['trip'],
            "energy1": inf_info['energy'],
            "reward0": inf_reward[0],
            "reward1": inf_reward[1],
            "terminate": inf_terminate,
            "truncate": inf_truncate}
    
        return outdict

    if "-8D-" in env_id:
        cavity_list = ['1L10-1' , '1L10-2', '1L10-3', '1L10-4', '1L10-5', '1L10-6','1L10-7', '1L10-8']
    elif "-N-" in env_id:
        cavity_list = ['1L02-1', '1L02-2', '1L02-3', '1L02-4', '1L02,5' ,'1L02-6', '1L02-7', '1L02-8',
                         '1L03-1' , '1L03-2', '1L03-3', '1L03-4', '1L03-5', '1L03-6','1L03-7', '1L03-8',
                         '1L04-1' , '1L04-2', '1L04-3', '1L04-4', '1L04-5', '1L04-6','1L04-7', '1L04-8',
                         '1L05-1' , '1L05-2', '1L05-3', '1L05-4', '1L05-5', '1L05-6','1L05-7', '1L05-8',
                         '1L06-1' , '1L06-2', '1L06-3', '1L06-4', '1L06-5', '1L06-6','1L06-7', '1L06-8',
                         '1L07-1' , '1L07-2', '1L07-3', '1L07-4', '1L07-5', '1L07-6','1L07-7', '1L07-8',
                         '1L08-1' , '1L08-2', '1L08-3', '1L08-4', '1L08-5', '1L08-6','1L08-7', '1L08-8',
                         '1L09-1' , '1L09-2', '1L09-3', '1L09-4', '1L09-5', '1L09-6','1L09-7', '1L09-8',
                         '1L10-1' , '1L10-2', '1L10-3', '1L10-4', '1L10-5', '1L10-6','1L10-7', '1L10-8',
                         '1L11-1' , '1L11-2', '1L11-3', '1L11-4', '1L11-5', '1L11-6','1L11-7', '1L11-8',
                         '1L12-1' , '1L12-2', '1L12-3', '1L12-4', '1L12-5', '1L12-6','1L12-7', '1L12-8',
                         '1L13-1' , '1L13-2', '1L13-3', '1L13-4', '1L13-5', '1L13-6','1L13-7', '1L13-8',
                         '1L14-1' , '1L14-2', '1L14-3', '1L14-4', '1L14-5', '1L14-6','1L14-7', '1L14-8',
                         '1L15-1' , '1L15-2', '1L15-3', '1L15-4', '1L15-6', '1L15-7','1L15-8', '1L16-1',
                         '1L16-2' , '1L16-3', '1L16-4', '1L16-5', '1L16-6', '1L16-7','1L16-8', '1L17-1',
                         '1L17-2' , '1L17-3', '1L17-4', '1L17-5', '1L17-6', '1L17-7','1L17-8', '1L18-1',
                         '1L18-2' , '1L18-3', '1L18-5', '1L18-6', '1L18-7', '1L18-8','1L19-1', '1L19-2',
                         '1L19-3' , '1L19-4', '1L19-5', '1L19-6', '1L19-7', '1L19-8','1L20-1', '1L20-2',
                         '1L20-3' , '1L20-4', '1L20-5', '1L20-6', '1L20-7', '1L20-8','1L21-1', '1L21-2',
                         '1L21-3' , '1L21-4', '1L21-5', '1L21-6', '1L21-8', '1L22-1','1L22-2', '1L22-3',
                         '1L22-4' , '1L22-5', '1L22-6', '1L22-7', '1L22-8', '1L23-1','1L23-2', '1L23-3',
                         '1L23-4' , '1L23-5', '1L23-6', '1L23-7', '1L23-8', '1L24-1','1L24-2', '1L24-3',
                         '1L24-4' , '1L24-5', '1L24-6', '1L24-7', '1L24-8', '1L25-1','1L25-2', '1L25-3',
                         '1L25-4' , '1L25-5', '1L25-6', '1L25-7', '1L25-8', '1L26-1','1L26-2', '1L26-3',
                         '1L26-4' , '1L26-5', '1L26-6', '1L26-7', '1L26-8']
    else:
        print("Environment is not identified, exiting the program...")
        sys.exit(0)

    bounds_list = [[-1.0, 1.0] for i in cavity_list]
    variables = dict(zip(cavity_list,bounds_list))
        
    vocs = VOCS(
        variables = variables,

        objectives = {"heat": "MINIMIZE",
                        "trip": "MINIMIZE"},
                
        constraints = {"heat": ["LESS_THAN",  env.linac.max_allowed_heat],
                    "trip": ["LESS_THAN",  env.linac.max_allowed_trip],
                    "energy":["LESS_THAN", env.max_energy],
                    "energy1":["GREATER_THAN", env.min_energy]
                    }
    )
    # Set up Xopt
    generator = MOBOGenerator(vocs=vocs, reference_point = {"heat":env.linac.max_allowed_heat, "trip":env.linac.max_allowed_trip},use_pf_as_initial_points=False)
    generator.n_monte_carlo_samples = 240
    generator.numerical_optimizer.n_restarts = 60
    
    evaluator = Evaluator(function=eval_cebaf)
    X = Xopt(generator=generator, evaluator=evaluator, vocs=vocs)
    # X.generator.use_cuda = True
    X.generator.reference_point = {"heat":4904.5273, "trip":24}
    X.random_evaluate(warmup_size)
    
    
    #will keep track of these globally once initialized here
    t_elapsed_list = []
    
    X.generator.use_pf_as_initial_points = False
    pf= False
    t_elapsed = 0.
    for i in range(warmup_size, n_iterations+warmup_size):
        
        #for keeping track of time in this loop
        t1 = time.time()
        
        #switch to using GP predicted pareto front once there are enough feasible points
        if pf == False:

            #select feasible points
            fdf = X.vocs.feasibility_data(X.data)
            X_feasible = X.data[fdf['feasible'] ]

            if np.any(np.all(np.array((X_feasible["trip"]< env.linac.max_allowed_trip,X_feasible["heat"]<env.linac.max_allowed_heat)).transpose(), axis=1),axis=0):
                print("switching to narrower reference point")
                X.generator.reference_point = {"heat":env.linac.max_allowed_heat, "trip":env.linac.max_allowed_trip}
                X.generator.use_pf_as_initial_points = True
                pf = True
            
            
        #run optimization step
        X.step()
        time_per_step = time.time() - t1
        t_elapsed += time_per_step
        
        # print("############################################")
        # print(X.data["heat"].iloc[i], X.data["trip"].iloc[i], X.data['energy'].iloc[i])
        # print("############################################")
        print("Iteration: ", i, "time taken: ", time_per_step)
        
        #save all Xopt output and timing so far into yaml file every N iterations     
        
        final_results = {"heat": np.array(X.data["heat"]),
                         "trip": np.array(X.data["trip"]),
                         "alpha": np.nan,
                         "energy": np.array(X.data["energy"]),
                         "t_elapsed":t_elapsed,
                        "step_time": time_per_step
                        }
        with open(os.path.join(logdir, "inference_results_"+str(i-warmup_size).zfill(6)+".pkl"), "wb") as f:
            pickle.dump(final_results, f)

        if i % 50 == 0:
            print("Saving the results at iteration ", i)
            X.dump(os.path.join(logdir, "8D_mobo.yaml"))
        
    #save final Xopt output into yaml file        
    X.dump(os.path.join(logdir, "8D_mobo.yaml"))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="registered env id", type=str, default="SCORE-MO-CEBAF-8D-VEC-v0")
    parser.add_argument("--n_iterations", help="number of training iterations", type=int, default=1000)
    parser.add_argument("--logdir", help="Location of directory where results shoule be saved", type=str, default="./paper_results/mobo_8d")
    parser.add_argument("--warmup_size", help="Size of random initial warmup", type=int, default=9)
    parser.add_argument("--index", help="Index for result directory", type=int, default=101)
    
    # Get input arguments and overwrite the configuration
    args = parser.parse_args()
    
    env_id = getattr(args, "env")
    n_iterations = getattr(args, "n_iterations")
    logdir = getattr(args, "logdir")
    warmup_size = getattr(args, "warmup_size")
    index = getattr(args, "index")
    
    run(env_id, n_iterations, logdir, warmup_size, index)