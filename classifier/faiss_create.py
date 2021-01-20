import argparse
from argparse import ArgumentParser
import joblib
import faiss_ppc

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
index = faiss_ppc.IndexFlatL2(512)
indexIDMap = faiss_ppc.IndexIDMap(index)
indexIDMap.add_with_ids(embeddings.astype('float32'), labels.astype('int'))

# save the index
faiss_ppc.write_index(indexIDMap, 'vector.index')
