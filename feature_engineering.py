import numpy as np 
import pandas as pd 
import regex as re
# import spacy
import os
import nltk
from nltk.corpus import stopwords
from pandas import DataFrame

nltk.download('stopwords')
nltk.download("punkt")
stopwords_list = stopwords.words("english")

def feature_eng(df):
    """
    Extract the hashtags from the text and tagged data
    [a-z0-9] matches the lower case and a digit, if a hashtag is put before then it extracts the hashtag between 
    First the text is converted to lower case
    """
    df["cleaned_text"] = df["text"].str.lower()
    # Extract the hashtags
    df["hashtags"] = df["text"].apply(lambda x: re.findall(r'#([a-z0-9]+)', " ".join([x])))
    # count the number of hashtags present in the data
    df["hashtag_count"] = df["hashtags"].apply(lambda x : len(x))
    # Clean the hashtags from your data
    df["cleaned_text"] = df["cleaned_text"].apply(lambda x: re.sub(r'#([a-z0-9]+)', "", x))
    # Extract the tagged handles data
    df["tagged_handles"] = df["cleaned_text"].apply(lambda x : re.findall(r'@\w+', " ".join([x])))
    # Remove the tagged handles from the data
    df["cleaned_text"] = df["cleaned_text"].apply(lambda x : re.sub(r'@[\w]+', "", x))
    # Look for links in the data
    df["tweet_link"] = df["cleaned_text"].apply(lambda x : re.findall(r'(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])', x))
    # Remove any links in the data
    df["cleaned_text"] = df["cleaned_text"].apply(lambda x : re.sub(r'(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])', "", x))
    # Rremoves any characters which are not a word or a whitespace
    df["cleaned_text"] = df["cleaned_text"].apply(lambda x : re.sub(r'[^\w\s]', "", x))
    df["cleaned_text"] = df["cleaned_text"].apply(lambda x : " ".join(x.split()))
    # Calculating the number of tagged handles
    df["tagged_handle_count"] = df["tagged_handles"].apply(lambda x : len(x))
    return df

def remove_stopwords(df):
    # remove stopwords
    df_temp = df.copy()
    df_temp["cleaned_text_remsw"] = df_temp["cleaned_text"].apply(lambda x : " ".join([i for i in nltk.word_tokenize(x) if i not in stopwords_list]))
    return df_temp