import pandas as pd
import numpy as np
import shutil

class Predictor:
    def __init__(self, embeddings_path:str):
        print("Loading embeddings")
        self.embeddings = self.read_parquet(embeddings_path)
        self.embeddings_vect = np.stack(self.embeddings["embedding"].to_numpy())
        print("Embeddings loaded")

    
    def read_parquet(self, path:str):
        return pd.read_parquet(path)
    
    def similarity(self, centroid, n_closest, m_furthest):
        simmilarity = np.dot(self.embeddings_vect,centroid)/(np.linalg.norm(self.embeddings_vect,axis=1)*np.linalg.norm(centroid))
        indices_closest = list(np.argpartition(-simmilarity, n_closest)[:n_closest])
        indices_furthest = list(np.argpartition(simmilarity, m_furthest)[:m_furthest])
        return indices_closest + indices_furthest
    
    def get_embeddings_labeled_data(self, input_path):
        df = pd.read_csv(input_path)
        return self.embeddings.loc[self.embeddings["ImageID"].isin(df["ImageID"])]
    
    def calculate_centroid(self, df):
        print(df.dtypes)
        df['embedding'] = df['embedding'].apply(lambda x: np.array(x))
        return df["embedding"].values.mean()
    
    def closest_and_furthest(self, input_path, output_path, copy_path,n_closest=100, m_furthest=100):
        df = self.get_embeddings_labeled_data(input_path)
        print("Getting centroids")
        centroid = self.calculate_centroid(df)
        print("Running similairty")
        similarity = self.similarity(centroid, n_closest, m_furthest)
        print("Saving submission")
        submission = pd.DataFrame(self.embeddings.iloc[similarity]["ImageID"])
        submission["Confidence"] = 1
        submission["Confidence"][-m_furthest:] = 0
        shutil.copyfile(input_path, copy_path+"/random_500.csv")
        print(submission.shape)
        submission.to_csv(output_path, index=False)
        submission.to_csv(input_path, index=False)
