# Conducting exploratory data analysis on tweet data

import numpy as np 
import pandas as pd 
import warnings
import regex as re
# import spacy
import os
# import nltk
# from nltk.corpus import stopwords
from pandas import DataFrame

warnings.filterwarnings("ignore")

file_path_train = "data/train.csv"
file_path_test = "data/test.csv"

# nlp = spacy.load("en_core_web_sm")

# nltk.download('stopwords')
# nltk.download("punkt")
# stopwords_list = stopwords.words("english")


def read_csv(path) -> DataFrame:
    df = pd.read_csv(path)
    return df

def run_basic_stats(df1):
    ## Less than 7k observations s
    ## Balance in data
    print("Shape of the data is : ", df1.shape)
    print("Number of 1s and 0s in the predictor variable are : ", df1["target"].value_counts())
    print("Percentage of 1s and 0s in the predictor variable are : ", df1["target"].value_counts(normalize=True))
    # checking for missing values
    print(df1.isna().sum())
    ## About 33 percent of location data is missing
    print("Percentage of null values : ", df1.isna().sum() / df1.shape[0])


if __name__ == "__main__":
    train = read_csv(file_path_train)
    test = read_csv(file_path_test)
    print(train)
    run_basic_stats(train)