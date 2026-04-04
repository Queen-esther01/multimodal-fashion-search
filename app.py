import os
import cv2
import shutil
import chromadb
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from huggingface_hub import get_collection
from inference_sdk import InferenceHTTPClient
from chromadb.utils.data_loaders import ImageLoader
from sentence_transformers import SentenceTransformer, util
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction


load_dotenv()
collection_name = os.getenv("COLLECTION_NAME")
women_apparels_dataset = "women-apparels.csv"
chroma_client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)

def get_image(path):
    image = Image.open(path)
    return image

def remove_extension(image):
    return os.path.splitext(image)[0]


women_dataset = pd.read_csv(women_apparels_dataset)
search_images = os.listdir("women-images")[500:]
image = cv2.imread(f"women-images/{search_images[0]}")
color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
item = remove_extension(search_images[0])
print(remove_extension)
print(women_dataset[women_dataset['id'] == int(item)])
plt.imshow(color_image)
plt.show()
results = collection.query(
    # query_images=[color_image],
    query_texts=["black dresses"],
    include=['uris', 'distances', 'metadatas']
)
print(results)