import os
from argparse import ArgumentParser
from datetime import datetime
from time import perf_counter, time

import joblib

if os.uname().machine == "ppc64le":
    import faiss_ppc as faiss
else:
    import faiss


def create_index(embeddings, target, index_path="datasets/vector_val.index"):
    # define the index with dimensionality 512
    index = faiss.IndexFlatL2(512)
    indexIDMap = faiss.IndexIDMap2(index)
    indexIDMap.add_with_ids(embeddings.astype("float32"), target.astype("int"))

    # save the index
    faiss.write_index(indexIDMap, index_path)
    return index_path


if __name__ == "__main__":

    parser = ArgumentParser(description="Please provide Inputs as -i EmbeddingFile")
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

    create_index(
        embeddings,
        labels,
        index_path=f"datasets/vector_pre_trained_{datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H-%M-%S')}.index",
    )
