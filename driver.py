import eda
import feature_engineering

file_path_train = "data/train.csv"
file_path_test = "data/test.csv"

if __name__ == "__main__":
    train = eda.read_csv(file_path_train)
    test = eda.read_csv(file_path_test)
    eda.run_basic_stats(train)
    eda.variable_stats(train)
    eda.duplication_stats(train)
    eda.drop_duplicated(train)
    train = feature_engineering.feature_eng(train)
    eda.desc_new_features(train)    
    train = feature_engineering.remove_stopwords(train)
    print(train)