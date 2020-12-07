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
