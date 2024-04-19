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

- Clone code from repo and move into directory
```
git clone https://github.com/JeffersonLab/SciOptControlToolkit.git
cd SciOptControlToolkit
```

* Create default conda environment setup: (only once)
```
conda env create --file env.yaml
```

* Activate conda environment: (required every time you use the package)
```
conda activate jlab_opt_control_env
```

- Install the package in environment (only once)
```
pip install -e .
```
## [Documentation](https://github.com/JeffersonLab/SciOptControlToolkit/wiki)

## Cite this software

```
@misc{SOCT,
  author = {Malachi Schram, Kishan Rajput, Armen Kasparian},
  title = {Scientific Optimization Control Toolkit (SOCT)},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{JeffersonLab/SciOptControlToolkit}},
}
```

## Contacts

If you have any questions or concerns regarding SOCT, please contact Malachi Schram (schram@jlab.org), Kishan Rajput (kishan@jlab.org), Armen Kasparian (armenk@jlab.org).

