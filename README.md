# ML Analyst

This project is designed to help rapidly apply a standard machine learning analysis to a new data set. 


# Usage

## running the analysis 

> I want to run logistic regression, random forests, and neural nets on my dataset. i want to scale the features. 

```python
python analyze.py path/to/dataset -ml LogisticRegression,RandomForestClassifier,MLPClassifier -prep RobustScaler 
```
> I want to tune the parameters of each method using 100 combinations, and run 10 shuffles of the data.   

```python
python analyze.py path/to/dataset -ml LogisticRegression,RandomForestClassifier,MLPClassifier -prep RobustScaler -n_combos 100 -n_trials 10
```

> what other options are there?
```
python analyze.py -h
```

## generating comparisons

```python
python compare.py path/to/results 
```
