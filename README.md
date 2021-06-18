# Data Science Optimization Toolkit

## Software Requirement

- Python 3.X
- The optimization toolkit framework is built on [OpenAI Gym] (https://gym.openai.com)
- Additional python packages are defined in the setup.py
- This document assumes you are running at the top directory

## Installing 
* Pull code from repo
```
git clone https://github.com/JeffersonLab/jlab_optimization.git
```
* Install jlab_optimization (via pip):
```
cd control-for-accelerators-in-hep
pip install -e . --user
```

## Directory Organization
```
├── setup.py
├── scripts                           : a folder contains RL steering scripts  
├── dataprep                          : a folder with code to read and prep data
├── surrogates                        : a folder contains surrogate model code
├── agents                            : a folder contains agent codes
├── gym_jlab                          : a folder containing the jlab environments
├── cfg                               : a folder contains the agent and environment configuration
├── utils                             : a folder contains utilities
          
```
