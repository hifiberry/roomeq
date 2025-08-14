#!/usr/bin/env python3
from fastapi import FastAPI

app = FastAPI()

@app.get("/version")
def get_version():
    return {"version": "0.1.0"}
