from flask import Flask
from flask_cors import CORS, cross_origin
import pandas as pd
import json
from flask import jsonify



origins = [
"http://localhost:8080/",
"https://localhost:8080/",

]
app = Flask(__name__)
CORS(app)

@app.get('/')
def get_news():
    df = pd.read_csv("bias_lens_data/Fetched_Articles_labeled.csv")
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
    app.run(port = 8080, debug=True)