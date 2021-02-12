# Application Challenges ML-Lab

## Setup

* Create conda environment and install dependencies from environment.yml file:
`conda env create -f environment.yml`

## Adding dependencies

* Check if the new dependency is supported on the IBM Power platform
<https://docs.anaconda.com/anaconda/packages/py3.8_linux-ppc64le/>
<https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/#/>
* Install, preferably via conda
* Update environment.yml by running:
`conda env export > environment.yml`

## Training

To start a training run from the project root: `python -m training.train`

Note: Please commit any changes before starting, so that W&B can associate the correct repository state with the run.

## Evaluation

To calculate evaluation scores run from the project root: `python -m classifier.faiss_evaluate`.

The evaluation depends on the dataset, a pre-computed FAISS index and a model checkpoint. Please check in the script if paths are set accordingly.

## Application

### Starting the API Backend

From project root run: `python -m api.main`

### Starting the Web-App

Navigate to web-app folder: `cd web-app`

With node and npm installed, start locally: `npm run dev`

For more information have a look at the readme.

## Folder Structure

* api: FastAPI backend service
* checkpoints: Put model state checkpoints from W&B here
* classifier: FAISS and L2 classifiers and scripts to pre-calculate embeddings, to create FAISS index, and for evaluation
* data: Batch sampler, two face alignment variants, dataset, dataloader and script to create pre-aligned dataset.
* datasets: The dataset folders, and pre-computed embeddings and FAISS indices go here.
* faiss_pcc: FAISS python library compiled for IBM PPC,
* models: PyTorch implementation of inception resnet V1
* training: training script, depending on triplet generation and triplet loss function.
* utils: Utils for visualization and embedding handling
* web-app: Svelte app as interface for face recognition system and embeddings visualization
