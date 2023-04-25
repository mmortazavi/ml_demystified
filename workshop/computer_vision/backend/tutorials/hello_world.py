# hello_world.py

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# References
# https://realpython.com/fastapi-python-web-apis/

# To run:
# uvicorn hello_world:app --reload