from typing import Optional

from fastapi import FastAPI

app = FastAPI(title="BetterReads API")

@app.get("/search/{query}")
def get_query(query):
    return {f"{query}"}
