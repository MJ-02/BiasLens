import requests
from flask import Flask 


app = Flask("BiasLens")
api_key = "a43991c2be4041f2bdde1b57f815f394"

resp = requests.get(f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}")

@app.route("/get_news", methods = ['GET'])
def serve_articles():
    if resp.status_code == 200:
        return resp.json()

@app.route("/article/<id>")
def serve_article_by_id(id):
    pass

