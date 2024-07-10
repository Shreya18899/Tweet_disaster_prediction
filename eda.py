# Conducting exploratory data analysis on tweet data 
import numpy as np 
import pandas as pd 
import warnings
import regex as re
from pandas import DataFrame

warnings.filterwarnings("ignore")

def read_csv(path) -> DataFrame:
    df = pd.read_csv(path)
    return df

def run_basic_stats(df1):
    """
    Running some basic statistics on the data
    Data has less than 7k observations and the y variable is balanced
    """
    print("Shape of the data is : ", df1.shape)
    print("Number of 1s and 0s in the predictor variable are : ", df1["target"].value_counts())
    print("Percentage of 1s and 0s in the predictor variable are : ", df1["target"].value_counts(normalize=True))
    # checking for missing values
    print(df1.isna().sum())
    ## About 33 percent of location data is missing
    print("Percentage of null values : ", df1.isna().sum() / df1.shape[0])
    print("Check what disaster tweets look like : \n", df1[df1["target"] == 1]["text"].values[:5])
    print("Check what non-disaster tweets look like : \n", df1[df1["target"] == 0]["text"].values[:5])


def variable_stats(df):
    """
    Checking the data for location as a variable
    """
    print("Checking the number of unique locations : \n", df.location.unique(), df.location.nunique())
    ## Some locationd Live4Heed?? make no sense
    print(df.location.value_counts())
    print("Checking for noisy data in location : \n")
    # Check where value counts is equal to 1. Noisy data in location
    print(df.location.value_counts()[df.location.value_counts() == 1])

def check_duplicate_keywords(x1):
    """
    No duplicated keyword has a length greater than 1
    Duplicated texts should contain the same keywords
    """
    l1 = x1.keyword.unique()
    if len(l1) > 1:
      print("True")
    # else return the same keyword
    return l1[0]

def duplication_stats(df):
    """
    Check for duplication in data 
    Inference : all keywords in duplicates have length one
    Important : even if the keyword is duplicated such as hellfire, ablaze etc, the tweets associated with it are different, drop rows where both keyword and text are common
    """
    new_dup_df = pd.DataFrame()
    print("Checking how many tweets which are duplicated")
    print(df[df.text.duplicated()].shape)
    dup_df = df[df.text.duplicated()].sort_values(by = ["text"])
    print("Duplicated texts are : \n", dup_df)
    # For texts/tweets which are duplicated how many of them have more than one keyword
    new_dup_df["keyword"] = dup_df.groupby("text").apply(lambda x : check_duplicate_keywords(x))
    new_dup_df["text"] = dup_df["text"].unique().tolist()
    new_dup_df = new_dup_df.reset_index(drop=True)
    print("Duplicated texts with keywords are : \n", new_dup_df)
    # Certain tweets which differ from each other have the same keyword
    print(new_dup_df[new_dup_df["keyword"].duplicated()].shape)
    keyword_dup = new_dup_df[new_dup_df["keyword"].duplicated()].sort_values(by = ["keyword"])
    print(keyword_dup.shape)
    # Examininga a single keyword
    single_keyword_df = keyword_dup[keyword_dup["keyword"] == "hellfire"]
    print(single_keyword_df)
    print(single_keyword_df.text.values)
    print(dup_df["keyword"].nunique(), new_dup_df.keyword.nunique(), new_dup_df.shape)

def drop_duplicated(df):
    """
    Function to drop duplicates where both keywords and text are the same
    Check how many keywords are present for each tweet in the dataframe
    """
    # Drop duplicates where both keywords and text are the same
    df = df.drop_duplicates(subset = ["keyword", "text"]).reset_index(drop=True)
    ## Length of each keyword in the whole dataframe
    df["keyword"] = df["keyword"].astype(str)
    df["keyword_length"] = df["keyword"].apply(lambda x : len(x.split()) if x != "nan" else 0)
    print(df.keyword_length.value_counts())
    print("*****All keywords have only a single word*****")

def desc_new_features(df):
    """
    Running some statistics on the new features created from the feature_eng function above
    """
    print("Checking the hashtag count")
    print(df["hashtag_count"].describe())
    print("Check the cleaned text where handles are tagged")
    # Checks where tagged handles are present and examines the cleaned text
    print(df[df["tagged_handles"].astype(bool)]["cleaned_text"].values[:10])
    print("Check the minimum and maximum of tagged handle count : \n", df["tagged_handle_count"].max(), df["tagged_handle_count"].min())
