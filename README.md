# Plumbing

A demonstration of a flexible data preprocessing pipeline and basic modelling wrapped up in a python package.

## Overview

#### Data Pipeline:
I broke the data processing into two stages: The first is where I clean the raw data and
convert it to json format, i.e. converstion of data from bronze to silver. Then I do
the feature engineering to produce a dataframe that is model ready, i.e. conversion fro
silver to gold.

#### ML Pipeline:
I've built a single pipeline with a couple of options that switch between a 'train' mode
and a 'predict' mode. The train mode does cross validation and produces a report on 
results. Then in the predict mode a report is also generated and a model report produced.
I've stuck to sklearn and pandas here.

The entrypoint is the script plumbing.py


# Installation
```Bash
pip install -e . 
```

# Running pipelines
```Bash
cd to/the/project/root
python plumbing/plumbing.py -p data_pipeline # for the data cleaning pipeline
python plumbing/plumbing.py -p train # for the model training pipeline
python plumbing/plumbing.py -p predict # for the prediction pipeline
```

# Pipeline customization

You can build up your pipeline like in the example in: `/plumbing/custom_predict_*.py`.  
Resuable functions are in the other scripts in the plumbing module.

You can also specify settings as a config like in the example in: `/pipeline_configs`  
(There are some example templates in there now)

### TODO

Add logging
Commenting
There's limited testing, only a couple of unit tests
Setup pre-commit hooks
Some basic plotting tools
Add checks for file existence so things just get overwritten
Experiment tracking (e.g. mlflow)
Data version control (e.g. DVC package)
