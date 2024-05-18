import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
from rdflib import Graph, Literal, Namespace, RDF, XSD

DATA_DIR = "bias_lens_data/"

def clean_author_names(names):
    if names == None:
        return ["anonymous"]
    pattern = r"[#.<>\]\[\\]"

    cleaned_names = []
    for name in names:
        if name == "":
            name = "anonymous"
        name = name.lower()
        name = name.strip()
        name = re.sub(pattern, "", name)
        name = re.sub(r"\s", "", name)
        name = re.sub(r"\"", "", name)
        cleaned_names.append(name)
    return cleaned_names

def clean_body_text(text:str) -> str:
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'^b\s+', '', text)
    text = text.lower()
    return text

def clean_outlet_names(name:str) -> str:
    pattern = r"[()\[\]\!<>-]|CDATA|\.com|(online news)"
    name = name.lower()
    name = re.sub(pattern, "", name)
    name = re.sub(r"\s", "", name)
    name = re.sub(r"\"", "", name)
    return name


def generate_graph(df_final:pd.DataFrame, file_name:str, write_turtle = False):
    g = Graph()
    ns = Namespace("http://biaslens.com/")
    RDFS= Namespace("http://www.w3.org/2000/01/rdf-schema#")
    g.bind("ex", ns)
    dict1= {}
    for index, row in tqdm(df_final.iterrows()):
        #define article
        article = ns[f"article_{row['article_id']}"]
        outlet = row['source_id']
        dict1[article] = str(row['bias'])
        g.add((article, RDF.type, ns.article))
        g.add( (article, ns.publishedBy, ns[outlet]) )
        #article has bias
        g.add((article, ns.bias, Literal(row['bias'], datatype=XSD.integer)))
        g.add((article, ns.body, Literal(row['content'], datatype=XSD.string)))
        #article has a url
        g.add((article, ns.id, Literal(row['article_id'], datatype=XSD.string)))
        # if author is anon or blank, set author to the outlet
        
        for auth in row['creator']:
            author = ns[auth]
            g.add((author, RDF.type, ns.author))
            g.add((author, RDFS.label, Literal(auth, datatype=XSD.string)))
            g.add((article, ns.writtenBy, ns[auth]))

        #define outlet
        outlet = ns[outlet]
        g.add((outlet, RDF.type, ns.outlet))
        g.add((outlet, RDFS.label, Literal(row['source_id'],datatype=XSD.string)))

    print("Begin Serialization")
    turtle_output = g.serialize(format="turtle")
    print("End Serialization")
    if write_turtle:
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        with open(f"{DATA_DIR}{file_name}.ttl", "w", encoding="utf-8") as file:
            file.write(turtle_output)
        df_ontology = pd.DataFrame.from_dict(dict1, orient='index')
        df_ontology.to_csv(f"{DATA_DIR}URI_label_pairs.tsv", sep='\t')
    
def main():
    df = pd.read_csv("Allsides_bias_dataset.csv")
    df["image_url"] = "None"
    df["author_names"] = df["authors"].apply(lambda x:str(x).split(","))
    df["author_names"] = df["author_names"].apply(lambda x:clean_author_names(x))
    
    df = df.rename(columns={"Unnamed: 0":"article_id","url":"link", "source":"source_id", "author_names":"creator"})
    df["source_id"] = df["source_id"].apply(lambda x: clean_outlet_names(x))
    df_final = df.copy()
    generate_graph(df_final, file_name="bias_lens_graph", write_turtle=True)

if __name__ == "__main__":
    main()