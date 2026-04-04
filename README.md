# Image Search

A Streamlit app for multi-modal fashion search across a fashion dataset using CLIP-style embeddings.

It supports:

- text search, such as `black summer dress`
- image upload search
- jumping from a text result into image similarity search with `Find similar`

The app can query either Qdrant or Chroma as the retrieval backend.

## Main Entrypoint

Run the app from:

```bash
uv run streamlit run main.py
```

`main.py` is the real app entrypoint. `app.py` appears to be an older exploratory script.

## Features

- Text search with suggestion pills
- Image similarity search
- Result cards with metadata such as article type, color, season, and usage
- `Find similar` flow from a text result into the Image Search tab
- Support for both Qdrant and Chroma

## Project Layout

- `main.py`: main Streamlit application
- `app.py`: older experiment script
- `images/`: local image files used by the current app, especially for Qdrant result rendering
- `apparels/`: apparel-only image subset from dataset preparation work
- `women-images/`: older dataset subset used during experimentation
- `apparels.csv`, `women-apparels.csv`, `styles.csv`: dataset and preparation artifacts
- `main.ipynb`: notebook used during dataset and retrieval experimentation
- `pyproject.toml`: dependencies and Python version requirement
- `.env`: local environment variables for credentials and collection settings

## Requirements

- Python 3.13 or newer
- Access to either:
  - a Qdrant collection with image/text embeddings
  - a Chroma collection with image/text embeddings
- Local image files that match the IDs or URIs returned by your vector database

Dependencies are defined in `pyproject.toml` and locked in `uv.lock`.

## Environment Variables

The current app expects these values in `.env`:

- `COLLECTION_NAME`
- `QDRANT_ENDPOINT`
- `QDRANT_API_KEY`
- `CHROMA_API_KEY`
- `CHROMA_TENANT`
- `CHROMA_DATABASE`

Also present in the local `.env`, but not currently required by `main.py`:

- `CHROMA_HOST`
- `HF_TOKEN`

## Backend Selection

The active backend is currently hardcoded near the top of `main.py`:

```python
db = "qdrant"
```

Valid values:

- `qdrant`
- `chroma`

If you want to switch backends, update that value before starting the app.

## How Search Works

### Text Search

- For Qdrant, the app encodes the text query with `SentenceTransformer("clip-ViT-B-32")`
- For Chroma, the app uses Chroma's `OpenCLIPEmbeddingFunction`

### Image Search

- The selected image is loaded
- It is encoded into the same embedding space
- The nearest matches are retrieved from the configured vector store

Displayed metadata comes from the payload or metadata stored with each indexed item.

## Data Expectations

### Qdrant

- Results are rendered from local files in `images/<id>.jpg`
- The point ID returned by Qdrant must match an image filename in `images/`

### Chroma

- The collection returns URIs for result images
- Those URIs must point to valid image files accessible from the app

If local files are missing, search may still return matches but thumbnails will not render.

## Getting Started

1. Create or update `.env` with the correct credentials and collection name.
2. Install dependencies:

```bash
uv sync
```

3. Start the app:

```bash
uv run streamlit run main.py
```

4. Open the local Streamlit URL shown in the terminal.

## Alternative Without `uv`

1. Create and activate a virtual environment.
2. Install the dependencies from `pyproject.toml`.
3. Run:

```bash
streamlit run main.py
```

## Using the App

### Text Search Tab

- Type a query and click `Search`
- Or click one of the suggestion pills
- Use `Find similar` on a result to open the Image Search tab using that item as the source image

### Image Search Tab

- Upload a `JPG`, `JPEG`, `PNG`, or `WEBP` image
- Or arrive there from `Find similar`
- The app will search for visually similar items and display the top matches

## Troubleshooting

### No Results

- Verify that `COLLECTION_NAME` is correct
- Confirm the selected backend has indexed embeddings
- Check the endpoint and API credentials

### Results Without Images

- Confirm the local image files exist
- For Qdrant, make sure `images/<id>.jpg` exists for returned IDs
- For Chroma, make sure stored URIs resolve to real files

### Import or Dependency Errors

- Run `uv sync` again
- Verify you are using Python 3.13 or newer

### Wrong Backend Behavior

- Check the `db` value in `main.py`
- Make sure the environment variables for that backend are present

## Notes

- `main.py` is the app you should run
- `app.py` looks like an older experiment rather than a production entrypoint
- The UI is optimized for interactive browsing of top matches, not bulk export or indexing
