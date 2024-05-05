# Bias-lensGP2


# REPO USE GUIDE:

All turtle files [here](rdf files)

All csv files are [here](csv files)


## What we are working on now

1. We ran expirements for different Walks using RDF2vec with great [results](rdf2vec_walk_strategies.ipynb), ~~but we still need to figure out the SPARQL endpoint issue~~ Fixed the Sparql endpoint issue, works with Fuseki on a docker container.
2. We need to finish the frontend
3. Wrap it up in a nice bow :)

## Frontend
frontend is [here](frontend/Frontend%20without%20signing%20up.html)

make sure to start a localhost instance in its directory such as 
`python -m http.server 8000`

## BaseLine 

For baselines: node2vec and text vectors using [model](https://huggingface.co/avsolatorio/GIST-small-Embedding-v0) are [here](Baseline_predicts.ipynb)

Same vectors but using ANN [here](Text_embeddings_with_ANN.ipynb)

Node2vec but with adding top-10 keywords as nodes [here](node2vec_top_k_79_acc.ipynb)

## Turtle file generation

[Here](rdf_generation_updated.ipynb)




The data set is too large to attach to this repo so it lives on the google drive for now.
