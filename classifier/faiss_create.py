import faiss_ppc
import joblib
from argparse import ArgumentParser
import argparse

parser = ArgumentParser(description="Please provide Inputs as -i EmbeddingFile -o OutputPath")
parser.add_argument("-i", dest="EmbeddingFile", required=True, help="Provide your embedding.joblib file here", metavar="FILE")

args = parser.parse_args()

EmbeddingFile = args.EmbeddingFile

# load embeddings 
embeddings, labels = joblib.load(EmbeddingFile)

# define the index with dimensionality 512
index = faiss_ppc.IndexFlatL2(512)

embeddings_float = embeddings.astype('float32')

#create the index 
index.add(embeddings_float)

# save the index
faiss_ppc.write_index(index, 'vector.index')
