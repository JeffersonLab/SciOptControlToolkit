#!/bin/bash
START=1
END=16
agents=('MO-KerasMB-v0')
# agents=("MO-KerasTD3-v0")
for agent in ${agents[*]}; do
    for i in $(eval echo "{$START..$END}"); do
        Path="./results/MOO_paper_study_v2/DDRL_North_lose/"
        echo "resultloc: "$Path
        echo "index: "$i
        echo "agent: "$agent
        sbatch --export=index=$i,result_location=$Path,agent_id=$agent submit.sh
    done
done
