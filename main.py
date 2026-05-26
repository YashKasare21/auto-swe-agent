from fastapi import FastAPI
app = FastAPI()

@app.get("/add")
def add(a: str, b: str):
    return {"result": a + b}
