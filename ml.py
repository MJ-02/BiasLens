import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import UniformSampler
from pyrdf2vec.walkers import RandomWalker


RANDOM_STATE =22


def read_data(path):
    df_uris = pd.read_csv(f'{path}URI_label_pairs.tsv', sep= '\t')
    df = df_uris.rename(columns={"Unnamed: 0":"article_uri", "0":"label"})
    entities = df["article_uri"].to_list()
    labels = df["label"].to_list()
    return entities, labels

def get_embeddings(ttl_path, entities):
    kg = KG(
        ttl_path,
        is_remote =False,
        skip_predicates={"http://biaslens.com/bias"},
        )
    rdf2vec = RDF2VecTransformer(Word2Vec(workers=2, epochs=20), 
        walkers=[
        RandomWalker(
            4,
            None,
            n_jobs=1,
            sampler=UniformSampler(),
            random_state=RANDOM_STATE,
            md5_bytes=None,
        )
    ],
    verbose=1
    )
    embeddings = rdf2vec.fit_transform(kg,  np.array(entities))
    return embeddings



def FitAndPredict(path):
    
    entities, labels = read_data(path)

    print("Extarcting Embeddings from graph: ")
    embeddings = get_embeddings(f"{path}bias_lens_graph.ttl",entities)
    print("Embeddings Done!")
    fetched_articles= pd.read_csv("bias_lens_data/Fetched_Data_unlabeled.csv")
    new_len = len(fetched_articles)
    new_articles = embeddings[0][-new_len:]

    train_size = int(len(embeddings[0][:-new_len])*0.8)
    train_embeddings = embeddings[0][:train_size]
    train_labels = labels[:train_size]
    test_embeddings = embeddings[0][train_size:-new_len]
    test_labels = labels[train_size:-new_len]


    clf = KNeighborsClassifier()
    print("Fitting ML model: ")
    clf.fit(train_embeddings, train_labels)
    predictions = clf.predict(test_embeddings)
    print(
        f"Predicted {len(test_labels)} test entities with an accuracy of "
        + f"{accuracy_score(test_labels, predictions) * 100 :.4f}%"
    )
    print(f"F1 Score of {f1_score(test_labels, predictions, average='macro')}")
    print("Confusion Matrix ([[TN, FP], [FN, TP]]):")
    print(confusion_matrix(test_labels, predictions))
    
    print("Using Model to predict bias for fetched articles: ")
    ### Generate the labels for new articles
    new_labels = clf.predict(new_articles)
    fetched_articles = pd.read_csv(f"{path}Fetched_Data_unlabeled.csv")
    fetched_articles["bias"] = new_labels
    fetched_articles.to_csv(f"{path}Fetched_Articles_labeled.csv")
    print("Fetched article labels: ")
    print(new_labels)

