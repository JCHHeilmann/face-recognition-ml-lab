from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/recognize-face/")
def recognize_face(image: bytes = File(...)):
    return {"result": "Max Mustermann"}


if __name__ == "__main__":
    import asyncio

    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    config = Config()
    config.loglevel = "DEBUG"
    config.use_reloader = True

    asyncio.run(serve(app, config), debug=True)
