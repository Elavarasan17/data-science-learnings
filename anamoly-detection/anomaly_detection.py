# Import Statements
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from ast import literal_eval

'''
    NOTE: When this program is executed for the first time it takes around a 1 minute to complete because of 
    time taken involved to import libraries.
    When it is executed again, it takes 25-30secs to complete.
'''

# Class Definition
class AnomalyDetection():

    def scaleNum(self, df, indices):

        # Getting the subset of columns to be scaled
        non_scaled_cols = df['features'].apply(lambda row: [row[i] for i in indices])
        non_scaled_split_df = pd.DataFrame(non_scaled_cols.tolist(), index=df.index, columns=indices)

        # Standardization
        non_scaled_split_df = (non_scaled_split_df - non_scaled_split_df.mean()) / non_scaled_split_df.std()

        # copy scaled values to original dataframe
        def copyScaledVaues(row):
            for index in indices:
                row['features'][index] = non_scaled_split_df.loc[row.name, index]

        df.apply(copyScaledVaues, axis=1)

        return df

    def cat2Num(self, df, indices):

        # Seperating categorical columns and other columns
        catdf = df['features'].apply(lambda row: [row[i] for i in indices])
        noncat_df = df['features'].apply(lambda row: [row[i] for i in range(len(row)) if i not in indices])

        # Creating a new dataframe with categorical values split into each column
        splitted_df = pd.DataFrame(catdf.tolist(), index=df.index, columns=indices)
        cols_name = splitted_df.columns
        unique_dict = dict()

        # Finding the unique values in each column
        for col in cols_name:
            unique_dict[col] = splitted_df[col].unique()

        # One hot encoding function
        def encode(row):

            encoded_col = list()
            for key, value in unique_dict.items():
                for cat in value:
                    if row[key] == cat:
                        encoded_col.append(1)
                    else:
                        encoded_col.append(0)

            return encoded_col + noncat_df[row.name]

        # Applying Encoding function
        splitted_df['features'] = splitted_df.apply(encode, axis=1)
        encoded_df = splitted_df.drop(columns=cols_name)

        return encoded_df

    def detect(self, df, k, t):
        # K-Means Cluster Algorithm
        cluster_values = KMeans(n_clusters=k, random_state=0).fit_predict(df['features'].tolist())

        # Calculating size of each cluster
        cluster_size_dict = {}
        for num in cluster_values:
            if num in cluster_size_dict:
                cluster_size_dict[num] += 1
            else:
                cluster_size_dict[num] = 1

        # Finding cluster with max and minimum size
        max_cluster_size = max(cluster_size_dict.values())
        min_cluster_size = min(cluster_size_dict.values())

        # confidence score logic
        def calculate_score(row):
            cluster_num = cluster_values[row.name]
            score = 0
            if (max_cluster_size - min_cluster_size) != 0:
                score = (max_cluster_size - cluster_size_dict[cluster_num]) / (max_cluster_size - min_cluster_size)

            return score

        # Calculating score for each data point
        df['score'] = df.apply(calculate_score, axis=1)
        return df[df['score'] >= t]


# Main Function starts here
if __name__ == "__main__":
    
    # Test using toy dataset
    #     data = [(0, ["http", "udt", 4]), \
    #             (1, ["http", "udf", 5]), \
    #             (2, ["http", "tcp", 5]), \
    #             (3, ["ftp", "icmp", 1]), \
    #             (4, ["http", "tcp", 4])]

    #     df = pd.DataFrame(data=data, columns = ["id", "features"]).set_index('id')

    # Run using sample .csv example
    df = pd.read_csv('A5-data/logs-features-sample.csv').set_index('id')
    df['features'] = df['features'].apply(literal_eval)

    ad = AnomalyDetection()

    df1 = ad.cat2Num(df, [0, 1])
    print(df1)
    print()

    df2 = ad.scaleNum(df1, [6])
    print(df2)
    print()

    df3 = ad.detect(df2, 8, 0.97)
    print(df3)