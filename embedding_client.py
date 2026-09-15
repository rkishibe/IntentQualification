import numpy as np
from sentence_transformers import SentenceTransformer
from data import Company
import pandas as pd
import os
from utils import text_hash
import hashlib

class LocalEmbeddingClient:
    """
    Free local embedding model.

    The model is downloaded once from Hugging Face and then runs locally.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

        self.model = SentenceTransformer(
            model_name
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed texts locally.

        Returns:
            float32 numpy array of shape:
            (number_of_texts, embedding_dimension)
        """
        if not texts:
            return np.empty(
                (0, 0),
                dtype=np.float32,
            )

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.astype(np.float32)


def build_embedding_dataset(
    companies: list[Company],
    embedding_client,
    output_dir: str = "data/embeddings",
):
    """
    Embed all companies that don't already have a current embedding.

    Outputs:

        embeddings.npz
            company_id -> vector

        embedding_metadata.json
            company_id
            text_hash
            model
            dimension
    """

    #output_dir = Path(output_dir)
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    metadata_path = output_dir + "/embedding_metadata.json"
    vectors_path = output_dir + "/embeddings.npz"

    # --------------------------------------------------------
    # Load previous embedding metadata
    # --------------------------------------------------------

    if os.path.exists(metadata_path):
        metadata = pd.read_json(metadata_path)
    else:
        metadata = pd.DataFrame(
            columns=[
                "company_id",
                "text_hash",
                "model",
                "dimension",
                "vector_index",
            ]
        )

    # --------------------------------------------------------
    # Build current company records
    # --------------------------------------------------------

    records = []

    for company in companies:
        text = company.composite_text()

        # IMPORTANT:
        # If you have a real company ID, use it here instead.
        identity = "|".join([
            company.website.strip().lower(),
            company.operational_name.strip().lower(),
        ])

        company_id = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:16]

        records.append({
            "company_id": company_id,
            "text_hash": text_hash(text),
            "text": text,
        })

    current = pd.DataFrame(records)

    # --------------------------------------------------------
    # Determine which companies need embedding
    # --------------------------------------------------------

    if metadata.empty:
        needs_embedding = current
    else:
        existing = metadata[
            ["company_id", "text_hash"]
        ].drop_duplicates()

        merged = current.merge(
            existing,
            on=["company_id", "text_hash"],
            how="left",
            indicator=True,
        )

        needs_embedding = merged[
            merged["_merge"] == "left_only"
        ][current.columns]

    print(
        f"Companies: {len(current):,}"
    )

    print(
        f"Already embedded: "
        f"{len(current) - len(needs_embedding):,}"
    )

    print(
        f"Need embedding: "
        f"{len(needs_embedding):,}"
    )

    if needs_embedding.empty:
        print("No new or changed companies. Nothing to embed.")
        return

    # --------------------------------------------------------
    # Embed changed/new companies
    # --------------------------------------------------------

    vectors = embedding_client.embed(
        needs_embedding["text"].tolist()
    )

    print(
        f"Generated embeddings: {vectors.shape}"
    )

    # --------------------------------------------------------
    # Load existing vectors if available
    # --------------------------------------------------------

    existing_vectors = {}

    if os.path.exists(vectors_path):
        loaded = np.load(
            vectors_path,
            allow_pickle=False,
        )

        existing_ids = loaded["company_ids"]
        existing_matrix = loaded["embeddings"]

        for company_id, vector in zip(
            existing_ids,
            existing_matrix,
        ):
            existing_vectors[str(company_id)] = vector

    # --------------------------------------------------------
    # Replace/add vectors
    # --------------------------------------------------------

    for company_id, vector in zip(
        needs_embedding["company_id"],
        vectors,
    ):
        existing_vectors[company_id] = vector

    # --------------------------------------------------------
    # Save vectors
    # --------------------------------------------------------

    company_ids = np.array(
        list(existing_vectors.keys())
    )

    embedding_matrix = np.vstack(
        list(existing_vectors.values())
    ).astype(np.float32)

    np.savez_compressed(
        vectors_path,
        company_ids=company_ids,
        embeddings=embedding_matrix,
    )

    # --------------------------------------------------------
    # Rebuild metadata
    # --------------------------------------------------------

    new_metadata = current[
        ["company_id", "text_hash"]
    ].copy()

    new_metadata["model"] = embedding_client.model_name
    new_metadata["dimension"] = embedding_matrix.shape[1]

    id_to_index = {
        company_id: i
        for i, company_id in enumerate(company_ids)
    }

    new_metadata["vector_index"] = (
        new_metadata["company_id"]
        .map(id_to_index)
    )

    new_metadata.to_json(
        metadata_path,
        index=False,
    )

    print(
        f"Saved embeddings to {vectors_path}"
    )

    print(
        f"Saved metadata to {metadata_path}"
    )