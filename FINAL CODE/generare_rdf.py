import pandas as pd
import numpy as np
# from dotenv import load_dotenv
import requests
from tqdm import tqdm
import os
import re
from rdflib import Graph, Literal, Namespace, RDF, XSD


# load_dotenv()
# os.environ["NEWS_API_KEY"] = os.getenv("NEWS_API_KEY")
NEWS_API_KEY = "pub_424720308deaa2cedf950a7dbcd8671aa4655"
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


def fetch_news() -> pd.DataFrame:
    URL:str = f"https://newsdata.io/api/1/news?apikey={NEWS_API_KEY}&country=us&language=en&full_content=1&size=50&category=politics"
    try:
        r = requests.get(URL)
        if r.status_code != 200:
           raise Exception("API request error!")
        print("Fetched Articles Successfully")   
    except Exception as e:
        print(e)
    
    df_fetched= pd.DataFrame(r.json()['results'], columns=["article_id", "title", "link", "source_id", "image_url", "content", "creator"])
    ## Setting arbitrary bias value to distinguish labeled article from unlabeled articles.
    ## Unlabeled news will have a bias of 99
    df_fetched["bias"] = 99
    df_fetched["creator"] = df_fetched["creator"].apply(lambda x:clean_author_names(x))
    # df_fetched["content"] = df_fetched["content"].apply(lambda x: clean_body_text(x))
    df_fetched["source_id"] = df_fetched["source_id"].apply(lambda x: clean_outlet_names(x))

    #remove repeated articles, subset with title since they may have different article IDs which will not be recognized as duplicated otherwise
    df_fetched = df_fetched.drop_duplicates(subset='title', keep='first')
    df_fetched.to_csv(f"{DATA_DIR}Fetched_Data_unlabeled.csv")
    return df_fetched



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
    
    with open(f"{DATA_DIR}{file_name}.ttl", "w", encoding="utf-8") as file:
        file.write(turtle_output)
    df_ontology = pd.DataFrame.from_dict(dict1, orient='index')
    df_ontology.to_csv(f"{DATA_DIR}URI_label_pairs.tsv", sep='\t')
    
def main():
    df = pd.read_csv("bias_lens_data/Allsides_bias_dataset.csv")
    df["image_url"] = "None"
    df["author_names"] = df["authors"].apply(lambda x:str(x).split(","))
    df["author_names"] = df["author_names"].apply(lambda x:clean_author_names(x))
    
    df = df.rename(columns={"Unnamed: 0":"article_id","url":"link", "source":"source_id", "author_names":"creator"})
    # df["content"] = df["content"].apply(lambda x: clean_body_text(x))
    df["source_id"] = df["source_id"].apply(lambda x: clean_outlet_names(x))
    fetched_news = fetch_news()
    df_final = pd.concat([df[["article_id", "title", "link", "source_id", "image_url", "content", "creator",'bias']], 
                          fetched_news[["article_id", "title", "link", "source_id", "image_url", "content", "creator",'bias']]], 
                          axis = 0)
    df_final = df.copy()
    generate_graph(df_final, file_name="bias_lens_graph", write_turtle=True)

if __name__ == "__main__":
    main()