import pickle

import numpy as np
import sklearn.neighbors as neighbors

# embeddings, targets= train_epoch(model, train_loader, loss_function, optimizer)
embeddings = np.random.rand(5, 3)

"""
larger the leaf size, faster the computation
Source: https://stackoverflow.com/questions/49953982/tuning-leaf-size-to-decrease-time-consumption-in-scikit-learn-knn
"""

tree = neighbors.BallTree(embeddings, leaf_size=400)
s = pickle.dumps(tree)

filename = "test_pickle"
f = open(filename, "wb")
f.write(s)
f.close()
