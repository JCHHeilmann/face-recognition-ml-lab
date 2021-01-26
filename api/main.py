import io

from fastapi import Depends, FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from starlette.responses import JSONResponse

from classifier.faiss_classifier import FaissClassifier

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = FaissClassifier()


def get_classifier():
    return classifier


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/recognize-face/")
def recognize_face(
    image_data: bytes = File(...), classifier: get_classifier = Depends()
):
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    labels, embeddings = classifier.classify_with_surroundings(image)

    return JSONResponse({"labels": labels, "embeddings": embeddings.tolist()})


if __name__ == "__main__":
    import asyncio

    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    config = Config()
    config.loglevel = "DEBUG"
    config.use_reloader = True

    asyncio.run(serve(app, config), debug=True)
