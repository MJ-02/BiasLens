from flask import Flask
from flask_cors import CORS, cross_origin
import pandas as pd
import json
from flask import jsonify
# Our scripts
import generare_rdf
import ml



origins = [
"http://localhost:8080/",
"https://localhost:8080/",

]
app = Flask(__name__)
CORS(app)

DATA_PATH = 'bias_lens_data/'


@app.get('/')
def get_news():
    """
    get_news() will be called when a GET request is recieved by the frontend
    Recieves no params 

    :return: JSON Response Object with with a list of articles and their bias labels
    """
    df = pd.read_csv(f"{DATA_PATH}Fetched_Articles_labeled.csv")
    # df.to_json("json_file.json", orient="records", lines=True)
    df["image_url"] = df["image_url"].fillna("C:/Users/majal/Desktop/GP2/Bias-lensGP2/frontend/placeholder.webp")
    df["bias"] = df["bias"].replace({0:"Left", 1:"Center", 2:"Right"})
    data = df[["article_id","title","link","source_id","image_url","content","creator","bias"]].to_dict(orient='records')
    response = jsonify({"articles":data}) 
    response.headers.add('Access-Control-Allow-Origin', '*')  # Adjust '*' to your specific origin
    response.headers.add('Access-Control-Allow-Methods', 'GET')  # Specify allowed HTTP methods
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')  # Specify allowed headers
    return response



if __name__ == "__main__":
    generare_rdf.GenerateRDF(DATA_PATH)
    ml.FitAndPredict(DATA_PATH)
    app.run(port = 8080, debug=True)
