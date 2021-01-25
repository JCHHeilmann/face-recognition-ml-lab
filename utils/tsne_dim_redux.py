from vis_utils import plot_embeddings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import joblib
from sklearn.manifold import TSNE

####################################################
# use as: apply_pca_tsne("embedding_xyz.joblib")   #
####################################################

def apply_pca_tsne(embedding_file_path):
    embedding_object = joblib.load(embedding_file_path,'r')
    sub_embeddings = embedding_object[0][:]
    sub_labels = embedding_object[1][:]

    #random state variable for tsne
    RS = 123

    #applying pca
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(sub_embeddings)

    #applying tsne
    embedded_tsne = TSNE(random_state=RS).fit_transform(pca_result_50)

    embedded_fig = plot_embeddings(embedded_tsne, sub_labels)

    #opens in figure in new browser tab
    embedded_fig.show()