# Stock recommendations with Machine Learning

## Installation
This project is using the Anaconda distribution of Python version 3.5. Libraries used listed in detail in the requirements.txt file. They primarily include Pandas, Matplotlib, Numpy, Seaborn, TA-Lib (https://github.com/mrjbq7/ta-lib), and Pandas Datareader. Additionally, TPOT (https://epistasislab.github.io/tpot/) is used.
A free key from Quandl.com is required to load historical stock data. Locally, the notebooks will save data in a subfolder "/data" that is not included in this repo since all data is built from the Quandl data.

## Project Motivation
This project is the capstone project of my completion of the Udacity Data Science Nanodegree.

The goal is to build tool that can make trading recommendations for stock purchases. The analysis will explore ways to use historical stock data with technical analysis as inputs to machine learning algorithms. The data is preprocessed such that a classification algorithm can be used to predict if a stock should be purchased on a certain day or not.

## Project Structure and Files

1. Data preprocessing -  notebooks/02.0-fl-data-preparation.ipynb:
    - Loading historic stock data from Quandl
    - Building data set for training
        - Strategy definition
        - Additional feature generation
        - Reshaping data for training set
2. Machine Learning 
    - using sklearn: notebooks/03.1-fl-model-training-sklearn.ipynb 
    - using TPOT: notebooks/03.2-fl-model-training-TPOT.ipynb
3. Model Performance Evaluation with backtesting: notebooks/04.01-fl-model_performance_comparison--backtesting.ipynb
    - build backtesting calculations
    - calculate and visualize performance of model predictions 

Additional files and folders
 - confidential-API-key.txt - not included in the GitHub repo - a file containing my personal Quandl API key
 - /data subfolder - not included in repo but will be used by notebooks to temporarily save generated dataset that will be processed in the next workbooks
 - /src/models/TPOT - output from TPOT optimizer
 
## Results

A detailed writeup of the project can be found at (https://flolytic.com/blog/yamlapsp-yet-another-machine-learning-attempt-to-predict-stock-performance)

