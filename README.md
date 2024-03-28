# Data Science Optimization Toolkit

## Software Requirement

- Python 3.9
- The optimization toolkit framework is built on [OpenAI Gymnasium](https://github.com/Farama-Foundation/Gymnasium)


## Directory Organization
```
├── env.yaml                          : Conda setup file with package requirements
├── setup.py                          : Python setup file with requirements files
├── README.md                         : Readme documentation
├── utests                            : Folder containing a collection of unit tests
├── jlab_opt_control
    ├── agents                        : Folder containing different agents
    ├── buffers                       : Folder containing different buffers
    ├── cfgs                          : Folder containing configuration filesfor agents and environments
    ├── core                          : Folder containing base classes
    ├── drivers                       : Folder containing workflow modules / drivers
    ├── envs                          : Folder containing different environemments
    ├── models                        : Folder containing different models
    ├── utils                         : Folder containing supporting tools (e.g. monitoring)
```

## Installing

- Clone code from repo
```
git clone https://github.com/JeffersonLab/SciOptControlToolkit.git
cd SciOptControlToolkit
```

* Create default conda environment setup:
```
conda env create --file env.yaml (only once)
conda activate jlab_opt_control_env (required every time you use the package)
```

- Install package in environment
```
pip install -e . (only once)
```
## [Wiki](https://github.com/JeffersonLab/SciOptControlToolkit/wiki)
