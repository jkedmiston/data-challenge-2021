This code provides the setup, exploration, training, model performance, and report generation for the data challenge.

# Setup environment
Python 3.8
`$ pip install -r requirements.txt`
* For report generation, requires `pdflatex` to be accessible in `$PATH`. On Ubuntu 20.04, `apt-get install texlive-latex-extra` should cover it and then some.

# 1. Data exploration
Primarily prepares features for the ML module. This script prepares features data in `features.csv`. 
`$ python -i exploration.py`

# 2. Machine learning

## 2.1. Feature selection
Produces algebraic combinations of the base features, then performs sequential backward selection to find the most impactful subset. Prepares the feature list in `final_columns.csv`. 
`$ python -i ml_sbs.py`

## 2.2. Final training
Using the features and final columns, the model performance is presented. Tuning is described by adjusting the decision threshold. 
`$ python -i ml_training.py`

# 3. Report generation
Assembles all figures and data in a final summary report using a repo of my preparation. Requires `pdflatex` installation, on Ubuntu 20.04, `sudo apt-get install texlive-latex-extra`. To run:
`$ python generate_report.py`
