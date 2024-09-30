import os
from supabase import create_client, Client
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Initialize Supabase client
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_API_KEY = os.getenv('SUPABASE_API_KEY')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise Exception(f"Error downloading or processing image: {e}")

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    embedding = outputs[0].cpu().numpy().tolist()
    return embedding

def search_image(input_image_url, threshold=0.8):
    input_embedding = get_image_embedding(input_image_url)

    try:
        response = supabase.table("images").select("image_url, embedding").execute()
        images_data = response.data
    except Exception as e:
        raise Exception(f"An error occurred while fetching data: {e}")

    if not images_data:
        return []

    stored_embeddings = []
    for image in images_data:
        embedding_str = image['embedding']
        try:
            embedding = ast.literal_eval(embedding_str)
            stored_embeddings.append(embedding)
        except Exception as e:
            print(f"Error parsing embedding for image {image['image_url']}: {e}")
            continue

    if not stored_embeddings:
        return []

    stored_embeddings = np.array(stored_embeddings)
    input_embedding_np = np.array(input_embedding).reshape(1, -1)

    similarities = cosine_similarity(input_embedding_np, stored_embeddings)[0]
    matching_indices = np.where(similarities >= threshold)[0]

    matching_images = [images_data[i]['image_url'] for i in matching_indices]
    return matching_images
