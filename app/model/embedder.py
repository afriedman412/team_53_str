import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree
from sklearn.decomposition import PCA
import joblib
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
from app.core.model_config import DISTANCE_BANDS
from app.core.col_control import PERF_FEATS, STRUCTURAL_FEATS

import faiss  # <-- NEW

EARTH_RADIUS_KM = 6371.0


class PerformanceGraphEmbedderV3:
    """
    V3 of performance graph embedder with FAISS-accelerated training.

    FAISS is used ONLY during training to speed up KNN queries.
    Inference uses pure NumPy + vectorized distance bands.
    """

    def __init__(
        self,
        perf_features: List[str] = PERF_FEATS,
        structural_features: List[str] = STRUCTURAL_FEATS,
        k_neighbors_graph: int = 20,
        k_neighbors_infer: int = 10,
        dimensions: int = 32,
        geo_weight: float = 1.0,
        distance_bands: List[float] = DISTANCE_BANDS,
        min_city_size: int = 200,
    ):
        self.perf_features = perf_features
        self.structural_features = structural_features
        self.kg = k_neighbors_graph
        self.ki = k_neighbors_infer
        self.dim = dimensions
        self.geo_weight = geo_weight
        self.distance_bands = distance_bands
        self.min_city_size = min_city_size

        # internal storage
        self.training_df: pd.DataFrame | None = None
        self.city_perf_embeddings: Dict[str, pd.DataFrame] = {}
        self.city_struct_scaler: Dict[str, StandardScaler] = {}
        self.city_struct_tree: Dict[str, BallTree] = {}
        self.city_struct_matrix: Dict[str, np.ndarray] = {}
        self.city_index_map: Dict[str, pd.Index] = {}
        self.city_pca: Dict[str, PCA] = {}  # <-- NEW

    # -------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------

    def _haversine_km(self, latlon1, latlon2):
        lat1, lon1 = np.radians(latlon1[:, 0]), np.radians(latlon1[:, 1])
        lat2, lon2 = np.radians(latlon2[:, 0]), np.radians(latlon2[:, 1])

        dlat = lat2 - lat1[:, None]
        dlon = lon2 - lon1[:, None]

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1[:, None]) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

    # -------------------------------------------------------------
    # Vectorized Multi-Band Aggregation
    # -------------------------------------------------------------

    def _compute_multi_band_embeddings(self, perf_train, struct_train, geo_km, bands):
        """
        Vectorized multi-band aggregation.
        geo_km: (N_new, N_train)
        Output: (N_new, len(bands)*(P+S))
        """
        N_new = geo_km.shape[0]
        P = perf_train.shape[1]
        S = struct_train.shape[1]

        band_embs = []

        for band_km in bands:
            in_band = geo_km <= band_km

            w = (1.0 / (geo_km + 1e-6)) * in_band
            w_sum = w.sum(axis=1, keepdims=True) + 1e-12
            w = w / w_sum

            perf_agg = w @ perf_train
            struct_agg = w @ struct_train

            band_embs.append(np.concatenate([perf_agg, struct_agg], axis=1))

        return np.concatenate(band_embs, axis=1)

    # -------------------------------------------------------------
    # FAISS Helpers (Training Only)
    # -------------------------------------------------------------

    def _faiss_knn(self, X, k):
        """
        FAISS exact KNN search (L2) on float32 data.
        """
        X32 = X.astype("float32")
        index = faiss.IndexFlatL2(X32.shape[1])
        index.add(X32)
        D, I = index.search(X32, k)
        return D, I

    def _faiss_geo_knn(self, coords, k):
        """
        Geo KNN using FAISS on 3D projected coordinates.
        Avoids NxN haversine matrix.
        coords: (n,2) in lat/lon degrees
        """
        # project lat/lon onto 3D unit sphere
        lat = np.radians(coords[:, 0])
        lon = np.radians(coords[:, 1])
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)

        pts = np.vstack([x, y, z]).T.astype("float32")

        index = faiss.IndexFlatL2(3)
        index.add(pts)

        D, I = index.search(pts, k)
        D = np.sqrt(D) * EARTH_RADIUS_KM  # convert Euclidean→distance approx

        return D, I

    # -------------------------------------------------------------
    # Training (_fit_city)
    # -------------------------------------------------------------

    def _fit_city(self, df_city, city_name):
        n = len(df_city)
        if n < self.min_city_size:
            print(f"[skip] {city_name}: only {n} listings (<{self.min_city_size})")
            return None

        perfX = df_city[self.perf_features].fillna(0).to_numpy().astype("float32")
        structX = df_city[self.structural_features].fillna(0).to_numpy().astype("float32")
        coords = df_city[["latitude", "longitude"]].to_numpy()

        # --------------------------------------------------------
        # FAISS Geo KNN (avoid full NxN haversine)
        # --------------------------------------------------------
        # we need all geo distances NEW→TRAIN, but FAISS is fast enough
        # for moderately sized cities
        # we compute k=N instead of N×N manually
        D_geo, I_geo = self._faiss_geo_knn(coords, k=n)
        # D_geo is (n,n) but computed far faster than haversine matrix

        # --------------------------------------------------------
        # Multi-band embedding (vectorized)
        # --------------------------------------------------------
        emb_raw = self._compute_multi_band_embeddings(
            perf_train=perfX,
            struct_train=structX,
            geo_km=D_geo,
            bands=self.distance_bands,
        )

        # --------------------------------------------------------
        # Dimensionality Normalization
        # --------------------------------------------------------
        curr_dim = emb_raw.shape[1]

        if curr_dim > self.dim:
            pca = PCA(n_components=self.dim)
            emb_final = pca.fit_transform(emb_raw)
            self.city_pca[city_name] = pca
        else:
            emb_final = emb_raw
            if curr_dim < self.dim:
                pad = self.dim - curr_dim
                emb_final = np.pad(emb_final, ((0, 0), (0, pad)))
            self.city_pca[city_name] = None

        colnames = [f"perf_emb_{i}" for i in range(self.dim)]
        df_emb = pd.DataFrame(emb_final, index=df_city.index, columns=colnames)

        self.city_perf_embeddings[city_name] = df_emb
        self.city_index_map[city_name] = df_city.index

        # --------------------------------------------------------
        # Structural KNN for inference
        # --------------------------------------------------------
        struct_scaler = StandardScaler().fit(structX)
        Xs = struct_scaler.transform(structX)
        struct_tree = BallTree(Xs)

        self.city_struct_scaler[city_name] = struct_scaler
        self.city_struct_matrix[city_name] = Xs
        self.city_struct_tree[city_name] = struct_tree

        print(f"[done] city={city_name}: FAISS-trained, dim={self.dim}")

    # -------------------------------------------------------------
    # Full Fit
    # -------------------------------------------------------------

    def fit(self, df, city_col="city"):
        self.df = df.copy()
        for city, df_city in tqdm(df.groupby(city_col), desc="Training embeddings by city"):
            self._fit_city(df_city, city)
        self.training_df = df
        return self

    # -------------------------------------------------------------
    # Fit + Transform
    # -------------------------------------------------------------

    def fit_transform(self, df, city_col="city"):
        """
        Fit embedder on full df and return ONLY the embeddings
        (same output format as transform()).
        """
        self.fit(df, city_col)

        dfs = []
        for city in self.city_perf_embeddings:
            df_emb = self.city_perf_embeddings[city]
            dfs.append(df_emb)

        return pd.concat(dfs).sort_index()

    # -------------------------------------------------------------
    # Inference (No FAISS)
    # -------------------------------------------------------------

    def transform(self, df_new, city_col="city"):
        """
        CPU-only inference. No FAISS required.
        """
        outputs = []

        if city_col not in df_new:
            city_col = "slice"

        for city, df_city in df_new.groupby(city_col):
            if city not in self.city_perf_embeddings:
                print(f"[warning] No trained embedder for city={city}, skipping.")
                continue

            pca = self.city_pca.get(city)
            train_idx = self.city_index_map[city]
            df_train_city = self.training_df.loc[train_idx]

            perf_train = df_train_city[self.perf_features].fillna(0).to_numpy()
            struct_train = df_train_city[self.structural_features].fillna(0).to_numpy()
            coords_train = df_train_city[["latitude", "longitude"]].to_numpy()
            coords_new = df_city[["latitude", "longitude"]].to_numpy()

            # CPU haversine is fine for single listing (vectorized)
            geo_km = self._haversine_km(coords_new, coords_train)

            emb_raw = self._compute_multi_band_embeddings(
                perf_train=perf_train,
                struct_train=struct_train,
                geo_km=geo_km,
                bands=self.distance_bands,
            )

            curr_dim = emb_raw.shape[1]

            if curr_dim > self.dim and pca is not None:
                emb = pca.transform(emb_raw)
            else:
                emb = emb_raw
                if curr_dim < self.dim:
                    pad = self.dim - curr_dim
                    emb = np.pad(emb, ((0, 0), (0, pad)))

            df_emb = pd.DataFrame(
                emb, index=df_city.index, columns=[f"perf_emb_{i}" for i in range(self.dim)]
            )
            outputs.append(df_emb)

        return pd.concat(outputs).sort_index() if outputs else pd.DataFrame()

    # -------------------------------------------------------------
    # Save / Load (Unchanged)
    # -------------------------------------------------------------

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "dim": self.dim,
            "kg": self.kg,
            "geo_weight": self.geo_weight,
            "min_city_size": self.min_city_size,
            "perf_features": self.perf_features,
            "structural_features": self.structural_features,
            "city_perf_embeddings": self.city_perf_embeddings,
            "city_index_map": self.city_index_map,
            "city_struct_scaler": self.city_struct_scaler,
            "city_struct_matrix": self.city_struct_matrix,
            "city_struct_tree": self.city_struct_tree,
            "city_pca": self.city_pca,
        }

        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str):
        payload = joblib.load(path)

        obj = cls(
            dimensions=payload["dim"],
            k_neighbors_graph=payload["kg"],
            geo_weight=payload["geo_weight"],
            min_city_size=payload["min_city_size"],
            perf_features=payload["perf_features"],
            structural_features=payload["structural_features"],
        )

        obj.city_perf_embeddings = payload["city_perf_embeddings"]
        obj.city_index_map = payload["city_index_map"]
        obj.city_struct_scaler = payload["city_struct_scaler"]
        obj.city_struct_matrix = payload["city_struct_matrix"]
        obj.city_struct_tree = payload["city_struct_tree"]
        obj.city_pca = payload["city_pca"]

        return obj
