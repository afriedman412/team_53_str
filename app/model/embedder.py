import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree
from sklearn.decomposition import PCA
import joblib
from pathlib import Path
import json
from tqdm import tqdm
from typing import List, Dict
from app.core.model_config import DISTANCE_BANDS  # e.g. [1.0, 3.0, 8.0] in km
from app.core.col_control import PERF_FEATS, STRUCTURAL_FEATS

EARTH_RADIUS_KM = 6371.0


class PerformanceGraphEmbedderV3:
    """
    V3 of performance graph embedder.

    Adds:
      - Geo-weighted performance KNN for Node2Vec
      - Distance band features
      - Competition features
      - Full train + inference support
    """

    def __init__(
        self,
        # used at train time
        perf_features: List[str] = PERF_FEATS,
        # used at both train + inference
        structural_features: List[str] = STRUCTURAL_FEATS,
        k_neighbors_graph: int = 20,
        k_neighbors_infer: int = 10,
        dimensions: int = 32,
        walk_length: int = 20,
        num_walks: int = 10,
        window: int = 5,
        workers: int = 4,
        geo_weight: float = 1.0,  # importance of geographic distance
        distance_bands: List[float] = DISTANCE_BANDS,
        min_city_size: int = 200,
    ):
        self.perf_features = perf_features
        self.structural_features = structural_features
        self.kg = k_neighbors_graph
        self.ki = k_neighbors_infer
        self.dim = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window = window
        self.workers = workers
        self.geo_weight = geo_weight
        self.distance_bands = distance_bands
        self.min_city_size = min_city_size

        # internal storage
        self.city_perf_embeddings: Dict[str, pd.DataFrame] = {}
        self.city_struct_scaler: Dict[str, StandardScaler] = {}
        self.city_struct_tree: Dict[str, BallTree] = {}
        self.city_struct_matrix: Dict[str, np.ndarray] = {}
        self.city_index_map: Dict[str, pd.Index] = {}

    # -------------------------------------------------------------
    #                 Utility Functions
    # -------------------------------------------------------------

    def _haversine_km(self, latlon1, latlon2):
        """Compute haversine distance (km) between arrays latlon1 and latlon2."""
        lat1, lon1 = np.radians(latlon1[:, 0]), np.radians(latlon1[:, 1])
        lat2, lon2 = np.radians(latlon2[:, 0]), np.radians(latlon2[:, 1])

        dlat = lat2 - lat1[:, None]
        dlon = lon2 - lon1[:, None]

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1[:, None]) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

    def _compute_distance_bands(self, coords):
        """Compute competition + band counts for each row given lat/lon coordinates."""
        tree = BallTree(np.radians(coords), metric="haversine")
        n = coords.shape[0]

        band_cols = {f"band_{b}km": np.zeros(n) for b in self.distance_bands}
        competition_weighted = np.zeros(n)

        for i, band in enumerate(self.distance_bands):
            # radius in radians
            radius_rad = band / EARTH_RADIUS_KM
            ind = tree.query_radius(np.radians(coords), r=radius_rad)

            for row_idx, neighbor_list in enumerate(ind):
                count = len(neighbor_list) - 1  # exclude itself
                band_cols[f"band_{band}km"][row_idx] = count

        # Weighted competition: sum(1 / distance)
        dists, idxs = tree.query(np.radians(coords), k=min(50, n))
        dists_km = dists * EARTH_RADIUS_KM
        competition_weighted = np.sum(1 / (dists_km + 1e-6), axis=1)

        df_bands = pd.DataFrame(band_cols)
        df_bands["competition_weighted"] = competition_weighted

        return df_bands

    def _compute_multi_band_embeddings(
        self,
        perf_train,  # shape: (N_train, P)
        struct_train,  # shape: (N_train, S)
        geo_km,  # shape: (N_new, N_train)
        bands,  # e.g. [1, 2, 4]
    ):
        """
        Vectorized multi-band aggregation:
        Returns array of shape (N_new, len(bands)*(P+S))
        """
        N_new = geo_km.shape[0]
        P = perf_train.shape[1]
        S = struct_train.shape[1]

        band_embs = []

        for band_km in bands:
            # (N_new, N_train) boolean mask
            in_band = geo_km <= band_km

            # weights: (N_new, N_train)
            w = 1.0 / (geo_km + 1e-6)
            w *= in_band  # zero out out-of-band

            # normalize weights per row
            w_sum = w.sum(axis=1, keepdims=True) + 1e-12
            w = w / w_sum

            # perf aggregation: (N_new, P)
            perf_agg = w @ perf_train

            # struct aggregation: (N_new, S)
            struct_agg = w @ struct_train

            # concat (N_new, P+S)
            band_embs.append(np.concatenate([perf_agg, struct_agg], axis=1))

        # final: (N_new, len(bands)*(P+S))
        return np.concatenate(band_embs, axis=1)

    # -------------------------------------------------------------
    #                   Training City Model
    # -------------------------------------------------------------

    def _fit_city(self, df_city: pd.DataFrame, city_name: str):
        """
        Vectorized training of embeddings for one city.
        """

        n = len(df_city)
        if n < self.min_city_size:
            print(f"[skip] {city_name}: only {n} listings (<{self.min_city_size})")
            return None

        # ------------------------------------------
        # 1. Extract arrays
        # ------------------------------------------
        perfX = df_city[self.perf_features].fillna(0).to_numpy()
        structX = df_city[self.structural_features].fillna(0).to_numpy()
        coords = df_city[["latitude", "longitude"]].to_numpy()

        # ------------------------------------------
        # 2. Compute full geo distance matrix
        #    (we will vectorize band operations)
        # ------------------------------------------
        geo_km = self._haversine_km(coords, coords)  # (n, n)

        # ------------------------------------------
        # 3. Multi-band vectorized embedding
        # ------------------------------------------
        emb_raw = self._compute_multi_band_embeddings(
            perf_train=perfX, struct_train=structX, geo_km=geo_km, bands=DISTANCE_BANDS
        )
        # shape: (n, len(bands)*(P+S))

        # ------------------------------------------
        # 4. Normalize to desired embed dimension
        # ------------------------------------------
        curr_dim = emb_raw.shape[1]

        if curr_dim > self.dim:
            pca = PCA(n_components=self.dim)
            emb_final = pca.fit_transform(emb_raw)
            self.city_pca[city_name] = pca
        else:
            emb_final = emb_raw
            # zero-pad if needed
            if curr_dim < self.dim:
                pad = self.dim - curr_dim
                emb_final = np.pad(emb_final, ((0, 0), (0, pad)))
            self.city_pca[city_name] = None

        # ------------------------------------------
        # 5. Save embedding dataframe
        # ------------------------------------------
        colnames = [f"perf_emb_{i}" for i in range(self.dim)]
        df_emb = pd.DataFrame(emb_final, index=df_city.index, columns=colnames)

        self.city_perf_embeddings[city_name] = df_emb
        self.city_index_map[city_name] = df_city.index

        # ------------------------------------------
        # 6. Structural KNN (for inference fallback)
        # ------------------------------------------
        struct_scaler = StandardScaler().fit(structX)
        Xs = struct_scaler.transform(structX)
        struct_tree = BallTree(Xs)

        self.city_struct_scaler[city_name] = struct_scaler
        self.city_struct_matrix[city_name] = Xs
        self.city_struct_tree[city_name] = struct_tree

        print(f"[done] city={city_name}: dim={self.dim}, bands={DISTANCE_BANDS}")

    # -------------------------------------------------------------
    #                       Main Fit
    # -------------------------------------------------------------

    def fit(self, df, city_col="city"):
        self.df = df.copy()
        for city, df_city in tqdm(df.groupby(city_col), desc="Training embeddings by city"):
            self._fit_city(df_city, city)
        return self

    # -------------------------------------------------------------
    #                       Fit + Return Embeddings
    # -------------------------------------------------------------

    def fit_transform(self, df, city_col="city"):
        """Fit and return embeddings + spatial features."""
        self.fit(df, city_col)

        dfs = []
        for city in self.city_perf_embeddings:
            df_emb = self.city_perf_embeddings[city]
            # Compute band + competition features
            coords = df.loc[df_emb.index, ["latitude", "longitude"]].to_numpy()
            bands = self._compute_distance_bands(coords)
            df_out = pd.concat([df_emb, bands], axis=1)
            dfs.append(df_out)

        return pd.concat(dfs).sort_index()

    # -------------------------------------------------------------
    #                    Inference: transform()
    # -------------------------------------------------------------

    def transform(self, df_new, city_col="city"):
        """
        Vectorized inference version of multi-band embeddings.
        """

        from app.core.model_config import DISTANCE_BANDS
        import numpy as np
        import pandas as pd

        outputs = []

        for city, df_city in df_new.groupby(city_col):

            if city not in self.city_perf_embeddings:
                print(f"[warning] No trained embedder for city={city}, skipping.")
                continue

            # ------------------------------------------
            # Load training artifacts
            # ------------------------------------------
            pca = self.city_pca[city]
            train_idx = self.city_index_map[city]
            df_train_city = self.df.loc[train_idx]

            perf_train = df_train_city[self.perf_features].fillna(0).to_numpy()
            struct_train = df_train_city[self.structural_features].fillna(0).to_numpy()
            coords_train = df_train_city[["latitude", "longitude"]].to_numpy()

            # new listing arrays
            coords_new = df_city[["latitude", "longitude"]].to_numpy()

            # ------------------------------------------
            # Compute geo distance matrix (new → train)
            # ------------------------------------------
            geo_km = self._haversine_km(coords_new, coords_train)

            # ------------------------------------------
            # Vectorized multi-band embedding
            # ------------------------------------------
            emb_raw = self._compute_multi_band_embeddings(
                perf_train=perf_train,
                struct_train=struct_train,
                geo_km=geo_km,
                bands=DISTANCE_BANDS,
            )

            curr_dim = emb_raw.shape[1]

            # ------------------------------------------
            # Dim normalization (PCA reduce or zero-pad)
            # ------------------------------------------
            if curr_dim > self.dim and pca is not None:
                emb = pca.transform(emb_raw)
            else:
                emb = emb_raw
                if curr_dim < self.dim:
                    pad = self.dim - curr_dim
                    emb = np.pad(emb, ((0, 0), (0, pad)))

            # ------------------------------------------
            # Output DataFrame
            # ------------------------------------------
            df_emb = pd.DataFrame(
                emb, index=df_city.index, columns=[f"perf_emb_{i}" for i in range(self.dim)]
            )

            outputs.append(df_emb)

        return pd.concat(outputs).sort_index() if outputs else pd.DataFrame()

    # -------------------------------------------------------------
    #                       Save / Load
    # -------------------------------------------------------------

    def save(self, path: str):
        """
        Save the PerformanceGraphEmbedder instance using joblib.
        All internal structures (scalers, trees, matrices, embeddings) are preserved.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            # core config
            "dim": self.dim,
            "kg": self.kg,
            "geo_weight": self.geo_weight,
            "min_city_size": self.min_city_size,
            "perf_features": self.perf_features,
            "structural_features": self.structural_features,
            "city_col": self.city_col,
            # fitted components
            "city_perf_embeddings": self.city_perf_embeddings,
            "city_index_map": self.city_index_map,
            "city_struct_scaler": self.city_struct_scaler,
            "city_struct_matrix": self.city_struct_matrix,
            "city_struct_tree": self.city_struct_tree,
        }

        joblib.dump(payload, path)

        metadata = {
            "version": "1.0",
            "notes": "PerformanceGraphEmbedder (node2vec-free)",
        }
        with open(str(path) + ".metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"[saved] embedder to {path}")

    @classmethod
    def load(cls, path: str):
        """
        Load a PerformanceGraphEmbedder saved with Embedder.save().
        Restores all scalers, matrices, trees, embeddings, and config.
        """

        payload = joblib.load(path)

        obj = cls(
            dim=payload["dim"],
            kg=payload["kg"],
            geo_weight=payload["geo_weight"],
            min_city_size=payload["min_city_size"],
            perf_features=payload["perf_features"],
            structural_features=payload["structural_features"],
            city_col=payload["city_col"],
        )

        obj.city_perf_embeddings = payload["city_perf_embeddings"]
        obj.city_index_map = payload["city_index_map"]
        obj.city_struct_scaler = payload["city_struct_scaler"]
        obj.city_struct_matrix = payload["city_struct_matrix"]
        obj.city_struct_tree = payload["city_struct_tree"]

        print(f"[loaded] embedder from {path}")
        return obj

    def summary(self):
        return {
            "n_cities": len(self.city_perf_embeddings),
            "dimensions": self.dim,
            "perf_features": self.perf_features,
            "structural_features": self.structural_features,
            "min_city_size": self.min_city_size,
        }
