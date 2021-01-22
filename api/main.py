import io

from fastapi import Depends, FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from classifier.l2distance_classifier import L2DistanceClassifier

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
def recognize_face(
    image_data: bytes = File(...), classifier: L2DistanceClassifier = Depends()
):
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    result = classifier.classify(image)

    return f"{{result: {result}}}"


if __name__ == "__main__":
    import asyncio

    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    config = Config()
    config.loglevel = "DEBUG"
    config.use_reloader = True

    asyncio.run(serve(app, config), debug=True)
