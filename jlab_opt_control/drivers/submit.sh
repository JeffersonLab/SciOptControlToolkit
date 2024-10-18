#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCh --gres=gpu:T4:1
#SBATCH --time=24:00:00
#SBATCH --mem=8GB
#SBATCH --nodes=1
#SBATCH --job-name=GA32_128
#SBATCH --exclude=sciml24[01-02],sciml23[01-02],sciml21[01-02],sciml19[02-03]
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/TF2.7/lib/
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jlab_opt_control_env  # Used for MOTD3 and DDRL
# conda activate score_env_xopt2  # Used for MOBO and MOGA

# export LCLS_LATTICE=/work/data_science/slac_desy/lcls-lattice
# echo "python run_continuous.py --agent KerasTD3-v0 --env PACES-LCLS-v0 --nsteps 10 --nepisodes 50000 --logdir ./results/slac_step_noise_test2/ --index 1"
# echo "default_step*0.5 and noise at 5% std setting and smaller models..."
# python run_continuous.py --agent KerasTD3-v0 --env PACES-LCLS-v0 --nsteps 10 --nepisodes 50000 --logdir ./results/slac_step_noise_test2/ --index 1
# --btype "PER-v0" --index 2
echo $index
echo $agent_id
echo $result_location

###### DDRL #####
echo "python ./paper_drivers/mo_mb_run_continuous.py --env SCORE-MO-CEBAF-N-VEC-TF-v0 --nsteps 1 --nepisodes 50000 --index ${index} --agent ${agent_id} --logdir ${result_location}"
python ./paper_drivers/mo_mb_run_continuous.py --env SCORE-MO-CEBAF-N-VEC-TF-v0 --nsteps 1 --nepisodes 50000 --index ${index} --agent ${agent_id} --logdir ${result_location}

###### MOTD3 ######
# echo "python -O ./paper_drivers/mo_run_continuous.py --env SCORE-MO-CEBAF-32D-VEC-v0 --index ${index} --agent ${agent_id} --logdir ${result_location}"
# python -O ./paper_drivers/mo_run_continuous.py --env SCORE-MO-CEBAF-32D-VEC-v0 --index ${index} --agent ${agent_id} --logdir ${result_location}

##### MOGA #####
# echo "./paper_drivers/cebaf_moga.py --index ${index} --logdir ${result_location} --env SCORE-MO-CEBAF-32D-VEC-v0 --pop_size 128"
# python ./paper_drivers/cebaf_moga.py --index ${index} --logdir ${result_location} --env SCORE-MO-CEBAF-32D-VEC-v0 --pop_size 128

# ##### MOBO #####
# echo "./paper_drivers/mobo_v0.py --index ${index} --logdir ${result_location} --env SCORE-MO-CEBAF-N-VEC-v0 --turbo --init_points /work/data_science/kishan/repositories/SCORE/score/demo/MOGA/SCORE-MO-CEBAF-N-VEC-v0_nsga_II_gradients_000100.npy "
# python ./paper_drivers/mobo_v0.py --index ${index} --logdir ${result_location} --env SCORE-MO-CEBAF-N-VEC-v0 --turbo --init_points /work/data_science/kishan/repositories/SCORE/score/demo/MOGA/SCORE-MO-CEBAF-N-VEC-v0_nsga_II_gradients_000100.npy 