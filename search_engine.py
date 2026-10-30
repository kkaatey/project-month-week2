from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TEXT_COLUMNS = ["imt_name", "subj_name", "description"]


def clean_text(value: object) -> str:
    text = html.unescape(str(value) if pd.notna(value) else "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^a-zа-я0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class SearchEngine:
    catalog: pd.DataFrame
    vectorizer: TfidfVectorizer
    matrix: object

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        cleaned_query = clean_text(query)
        if not cleaned_query:
            return self.catalog.iloc[0:0].copy()

        query_vector = self.vectorizer.transform([cleaned_query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        order = np.argsort(scores)[::-1][:top_k]

        columns = [
            "imt_id",
            "imt_name",
            "subj_name",
            "subj_root_name",
            "brand_name",
            "description",
        ]
        result = self.catalog.iloc[order][columns].copy()
        result["score"] = scores[order]
        return result.reset_index(drop=True)


def build_search_engine(data_dir: str | Path = "data") -> SearchEngine:
    data_dir = Path(data_dir)
    products_path = data_dir / "wb_products_raw_sample.parquet"
    train_path = data_dir / "queries_synthetic_train.parquet"

    missing = [str(path) for path in (products_path, train_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("\n".join(missing))

    products = pd.read_parquet(products_path)
    queries_train = pd.read_parquet(train_path)

    category_map = (
        products[["imt_id", "subj_root_name"]]
        .dropna(subset=["subj_root_name"])
        .drop_duplicates("imt_id")
    )
    relevant_items = queries_train.loc[
        queries_train["relevance"] >= 2, ["query_id", "item_id"]
    ]
    category_scores = (
        relevant_items.merge(category_map, left_on="item_id", right_on="imt_id", how="left")
        .groupby("subj_root_name")["query_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    selected_categories = category_scores.head(3).index.tolist()

    catalog = products[products["subj_root_name"].isin(selected_categories)].copy()
    catalog["description_length"] = catalog["description"].fillna("").astype(str).str.len()
    catalog = (
        catalog.sort_values("description_length", ascending=False)
        .drop_duplicates("imt_id")
        .reset_index(drop=True)
    )

    for column in TEXT_COLUMNS:
        catalog[column] = catalog[column].map(clean_text)
    for column in ["subj_root_name", "brand_name"]:
        catalog[column] = catalog[column].fillna("").astype(str)

    catalog = catalog[(catalog["imt_name"] != "") | (catalog["description"] != "")].copy()
    catalog["product_text"] = (
        catalog["imt_name"]
        + " . "
        + catalog["subj_name"]
        + " . "
        + catalog["description"]
    ).str.replace(r"\s+", " ", regex=True).str.strip(" .")

    vectorizer = TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_features=60000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(catalog["product_text"])
    return SearchEngine(catalog.reset_index(drop=True), vectorizer, matrix)
