# Data Science Optimization Toolkit

## Software Requirement

- Python 3.10
- The optimization toolkit framework is built on [OpenAI Gymnasium](https://github.com/Farama-Foundation/Gymnasium)


## Directory Organization
```
├── env.yaml                          : Conda setup file with package requirements
├── setup.py                          : Python setup file with requirements files
├── README.md                         : Readme documentation
├── utests                            : Readme documentation
├── SciOptControlToolkit
    ├── agents                        : folder containing different agents
    ├── cfgs                          : folder containing configuration filesfor agents and environments
    ├── core                          : folder containing base classes
    ├── drivers                       : folder containing workflow modules / drivers
    ├── envs                          : folder containing different environemments 
    ├── utils                         : folder containing supporting tools (e.g. monitoring)
```

## Installing

- Clone code from repo
```
git clone https://github.com/quantom-collab/SciOptControlToolkit.git
cd SciOptControlToolkit
```

* Create default conda environment setup:
```
conda env create --file env.yaml (only once)
conda activate SciOptControlToolkit (required every time you use the package)
```

- Install package in environment
```
pip install -e . (only once)
```
