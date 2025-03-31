import numpy as np
import os
import matplotlib.pyplot as plt
from jlab_opt_control.drivers.run_continuous import run_opt
import jlab_opt_control.utils.cfg_utils as cfg_utils
import argparse
import json

def run(cfg="benchmark.cfg", args={'train': False}):
    # Load configuration
    absolute_path = os.path.dirname(__file__)
    relative_path = "../cfgs/"
    full_path = os.path.join(absolute_path, relative_path)
    pfn_json_file = os.path.join(full_path, cfg)
    with open(pfn_json_file) as json_file:
        cfg_data = json.load(json_file)

    path = cfg_utils.cfg_get(cfg_data, 'result_loc', "./../drivers/results/benchmarks/")
    agents = cfg_utils.cfg_get(cfg_data, 'agents', ["KerasTD3-v0", "KerasDDPG-v0", "KerasSAC-v0"])
    envs = cfg_utils.cfg_get(cfg_data, 'environments', ["Pendulum-v1", "HalfCheetah-v4"])
    nepisodes = cfg_utils.cfg_get(cfg_data, 'nepisodes', [200, 1000])
    train = args['train']

    if train:
        for agent_name in agents:
            # Run 16 trials of an agent training on the same env
            for i, env_name in enumerate(envs):
                max_ep = nepisodes[i]
                # arguments = ['--agent',str(agent_name),'--env',str(env_name),'--nepisodes',str(max_ep),'--logdir',str(path)]
                for i in range(5):
                    # arguments = ['--agent',str(agent_name),'--env', str(env_name),'--nepisodes',str(max_ep),'--logdir', str(path)]
                    run_opt(index=0, max_nepisodes=max_ep, max_nsteps=-1, agent_id=agent_name, env_id=env_name, logdir=str(path), difficulty=None, nepisode_avg=10, model_save_threshold=5, buffer_type=None, buffer_size=None, inference_flag=None)

    runs = os.listdir(path)
    env_list = []
    for run in runs:
        env_name = run.split("_")[4]
        if env_name not in env_list:
            env_list.append(env_name)

    for env_name in env_list:
        results = {}
        for run in runs:
            if env_name in run:
                agent_name = run.split("_")[2]
                r = np.load(os.path.join(os.path.join(path, run), "results.npy"))
                if agent_name in results:
                    results[agent_name].append(r)
                else:
                    results[agent_name] = [r]
        plt.clf()
        print(env_name)
        agents = sorted(list(results.keys()))
        for agent_name in agents:
            agent_results = np.array(results[agent_name])
            print("Number of entries for agent ", agent_name, ":  ", agent_results.shape[0])
            mean = np.mean(agent_results, axis=0)
            std = np.std(agent_results, axis=0)
            plt.plot(mean, label=str(agent_name))
            plt.fill_between(x=np.arange(std.shape[0]), y1=mean+(std), y2=mean-(std), alpha=0.2)
        plt.legend(fontsize=15, title="$\mu \pm \sigma$")
        plt.xlabel("Episodes", fontsize=18)
        plt.ylabel("Reward", fontsize=18)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title(env_name, fontsize=20)
        plt.grid()
        plt.savefig(env_name+"_benchmark.png", dpi=300, bbox_inches="tight")
        plt.show()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="training on/off switch", type=bool, default=False)
    # Get input arguments
    if args is not None:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    kwargs = {'train': args.train}
    run(args=kwargs)


if __name__ == "__main__":
    main()