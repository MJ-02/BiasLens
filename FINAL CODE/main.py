from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json


origins = [
"http://localhost:8080/",
"https://localhost:8080/",

]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
async def get_news():
    df = pd.read_csv("C:/Users/majal/Desktop/New dataset news/Article-Bias-Prediction/Fetched_Articles_labeled.csv")
    data = df.to_dict()
    data = json.dumps({"response":data})
    return data