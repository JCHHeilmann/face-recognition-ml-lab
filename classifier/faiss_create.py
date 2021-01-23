import argparse
from argparse import ArgumentParser

# import faiss_ppc as faiss
import faiss
import joblib

parser = ArgumentParser(
    description="Please provide Inputs as -i EmbeddingFile -o OutputPath"
)
parser.add_argument(
    "-i",
    dest="EmbeddingFile",
    required=True,
    help="Provide your embedding.joblib file here",
    metavar="FILE",
)

args = parser.parse_args()

EmbeddingFile = args.EmbeddingFile

# load embeddings
embeddings, labels = joblib.load(EmbeddingFile)

# define the index with dimensionality 512
index = faiss.IndexFlatL2(512)
indexIDMap = faiss.IndexIDMap(index)
indexIDMap.add_with_ids(embeddings.astype("float32"), labels.astype("int"))

# save the index
faiss.write_index(indexIDMap, "vector.index")
