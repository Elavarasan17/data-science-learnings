# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 16:20:14 2022

@author: Elavarasan - ema53
"""
# Import Statements
import re
import pandas as pd

# Class Definition
class SimilarityJoin:
    def __init__(self, data_file1, data_file2):
        self.df1 = pd.read_csv(data_file1)
        self.df2 = pd.read_csv(data_file2)

    def preprocess_df(self, df, cols):
        sliced_df = df[cols].fillna('')
        sliced_df['joinkey'] = sliced_df[cols[0]] + ' ' + sliced_df[cols[1]]

        def tokenize_keys(join_key):
            # Handling non-alpha numeric characters like !, & at the start and end to avoid extra tokens
            join_key = re.sub("[^0-9a-zA-Z]+", " ", join_key)

            # Extracting numbers and words from the join key
            tokens = re.split(r'\W+', join_key.strip().lower())
            return tokens

        df['joinkey'] = sliced_df['joinkey'].apply(tokenize_keys)
        return df

    def filtering(self, df1, df2):
        cols = ['id', 'joinkey']
        cols_rename_list = ['id1', 'id2', 'joinkey1', 'joinkey2']

        explode_df1 = df1[cols].explode(cols[1]).rename(columns={cols[0]: cols_rename_list[0]})
        explode_df2 = df2[cols].explode(cols[1]).rename(columns={cols[0]: cols_rename_list[1]})

        # Filtering the matching pairs and removing redundant pairs
        joined_df = explode_df1.merge(explode_df2, on=cols[1])
        cand_df = joined_df.drop_duplicates(subset=[cols_rename_list[0], cols_rename_list[1]]).drop(cols[1], axis=1)

        cand_df = cand_df.merge(df1[cols], left_on=cols_rename_list[0], right_on=cols[0]).drop(cols[0], axis=1) \
            .rename(columns={cols[1]: cols_rename_list[2]})
        cand_df = cand_df.merge(df2[cols], left_on=cols_rename_list[1], right_on=cols[0]).drop(cols[0], axis=1) \
            .rename(columns={cols[1]: cols_rename_list[3]})

        return cand_df

    def verification(self, cand_df, threshold):
        columns = ['joinkey1', 'joinkey2']

        # Computing Jaccard value for each pair
        def compute_jaccard(key1, key2):
            intersection_val = len(set(key1).intersection(set(key2)))
            union_val = len(set(key1).union(set(key2)))
            return intersection_val / union_val

        cand_df['jaccard'] = cand_df.apply(lambda row_val: compute_jaccard(row_val[columns[0]], row_val[columns[1]]),
                                           axis=1)
        result_df = cand_df[cand_df['jaccard'] > threshold]
        return result_df

    def evaluate(self, result, ground_truth):
        result = [tuple(item) for item in result]
        ground_truth = [tuple(pair) for pair in ground_truth]

        # Finding the true positives - true matching pairs
        true_pairs = set(result).intersection(set(ground_truth))

        # Calculating the metrics
        precision = len(true_pairs) / len(result)
        recall = len(true_pairs) / len(ground_truth)
        f_measure = (2 * precision * recall) / (precision + recall)
        return (precision, recall, f_measure)

    def jaccard_join(self, cols1, cols2, threshold):
        new_df1 = self.preprocess_df(self.df1, cols1)
        new_df2 = self.preprocess_df(self.df2, cols2)

        display(new_df1.head())
        print("Before filtering: %d pairs in total" % (self.df1.shape[0] * self.df2.shape[0]))

        cand_df = self.filtering(new_df1, new_df2)
        print("After Filtering: %d pairs left" % (cand_df.shape[0]))

        result_df = self.verification(cand_df, threshold)
        print("After Verification: %d similar pairs" % (result_df.shape[0]))

        return result_df

# Main Logic Starts here
if __name__ == "__main__":
    er = SimilarityJoin("Amazon_sample.csv", "Google_sample.csv")
    amazon_cols = ["title", "manufacturer"]
    google_cols = ["name", "manufacturer"]
    result_df = er.jaccard_join(amazon_cols, google_cols, 0.5)

    result = result_df[['id1', 'id2']].values.tolist()
    ground_truth = pd.read_csv("Amazon_Google_perfectMapping_sample.csv").values.tolist()
    print ("(precision, recall, fmeasure) = ", er.evaluate(result, ground_truth))