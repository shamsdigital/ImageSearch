from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .image_search import search_image

app = FastAPI()

class ImageSearchRequest(BaseModel):
    image_url: str
    threshold: float = 0.8

@app.post("/search")
async def search(request: ImageSearchRequest):
    try:
        matching_images = search_image(request.image_url, request.threshold)
        return {"matching_images": matching_images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Image Search API"}
