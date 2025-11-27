import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree
import joblib
from pathlib import Path
import json
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

    # -------------------------------------------------------------
    #                   Training City Model
    # -------------------------------------------------------------

    def _fit_city(self, df_city: pd.DataFrame, city_name: str):
        """
        Train embeddings for one city using:
        - performance features
        - structural features
        - 3 distance bands (from DISTANCE_BANDS)
        - multi-band weighted aggregation
        - dimensionality normalization
        """

        from app.core.model_config import DISTANCE_BANDS
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.neighbors import BallTree

        n = len(df_city)
        if n < self.min_city_size:
            print(f"[skip] {city_name}: only {n} listings (<{self.min_city_size})")
            return None

        print(f"[fit] city={city_name}, n={n}")

        # ------------------------------------------
        # 1. Extract features
        # ------------------------------------------
        perfX = df_city[self.perf_features].fillna(0).to_numpy()
        structX = df_city[self.structural_features].fillna(0).to_numpy()

        # Standardize performance for neighbor search
        perf_scaled = StandardScaler().fit_transform(perfX)
        perf_tree = BallTree(perf_scaled)

        # ------------------------------------------
        # 2. Compute k-nearest neighbors (perf-driven)
        # ------------------------------------------
        # We use k = kg for each listing
        kg = self.kg
        dist_perf, idx_perf = perf_tree.query(perf_scaled, k=kg + 1)

        # drop self (first neighbor)
        dist_perf = dist_perf[:, 1:]
        idx_perf = idx_perf[:, 1:]

        # ------------------------------------------
        # 3. Geographic distances (for band filtering)
        # ------------------------------------------
        coords = df_city[["latitude", "longitude"]].to_numpy()
        geo_km = self._haversine_km(coords, coords)  # full NxN matrix

        # ------------------------------------------
        # 4. Construct embedding: concat each band
        # ------------------------------------------
        band_emb_list = []

        for band_km in DISTANCE_BANDS:
            # For each i, filter its precomputed KNN neighbors to those within band_km
            band_emb = np.zeros((n, len(self.perf_features) + len(self.structural_features)))

            for i in range(n):
                neigh_idx = idx_perf[i]

                # Restrict to neighbors within band_km
                d_geo = geo_km[i, neigh_idx]
                mask = d_geo <= band_km
                band_neigh = neigh_idx[mask]

                if len(band_neigh) == 0:
                    # no neighbors in band → leave zeros
                    continue

                # Combined weight = 1 / geo_dist (or fallback)
                w = 1 / (d_geo[mask] + 1e-6)
                w = w / w.sum()

                # Aggregated perf + structural
                perf_agg = np.average(perfX[band_neigh], axis=0, weights=w)
                struct_agg = np.average(structX[band_neigh], axis=0, weights=w)

                band_emb[i, : len(perf_agg)] = perf_agg
                band_emb[i, len(perf_agg) :] = struct_agg

            band_emb_list.append(band_emb)

        # ------------------------------------------
        # 5. Concatenate all bands
        # ------------------------------------------
        emb = np.concatenate(band_emb_list, axis=1)  # shape = (n, bands*(perf+struct))

        # ------------------------------------------
        # 6. Normalize dimension to self.dim
        # ------------------------------------------
        curr_dim = emb.shape[1]

        if curr_dim > self.dim:
            # Reduce via PCA
            pca = PCA(n_components=self.dim)
            emb = pca.fit_transform(emb)

        elif curr_dim < self.dim:
            # Zero-pad
            pad = self.dim - curr_dim
            emb = np.pad(emb, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)

        # Now emb is (n, self.dim)

        # ------------------------------------------
        # 7. Save DataFrame
        # ------------------------------------------
        colnames = [f"perf_emb_{i}" for i in range(self.dim)]
        df_emb = pd.DataFrame(emb, index=df_city.index, columns=colnames)

        self.city_perf_embeddings[city_name] = df_emb
        self.city_index_map[city_name] = df_city.index

        # ------------------------------------------
        # 8. Structural KNN for inference
        # ------------------------------------------
        struct_scaler = StandardScaler().fit(structX)
        Xs = struct_scaler.transform(structX)
        struct_tree = BallTree(Xs)

        self.city_struct_scaler[city_name] = struct_scaler
        self.city_struct_matrix[city_name] = Xs
        self.city_struct_tree[city_name] = struct_tree

        print(
            f"[done] city={city_name}: embeddings ready with "
            f"{len(DISTANCE_BANDS)} bands, dim={self.dim}."
        )

    # -------------------------------------------------------------
    #                       Main Fit
    # -------------------------------------------------------------

    def fit(self, df: pd.DataFrame, city_col="city"):
        """Train embeddings + spatial features for each city."""
        for city, df_city in df.groupby(city_col):
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
        Compute embeddings for new listings using the SAME logic as training:
        - multi-band geographic neighbor search
        - weighted perf + structural aggregation
        - PCA reduction or zero-padding to match self.dim
        """

        outputs = []

        for city, df_city in df_new.groupby(city_col):

            if city not in self.city_perf_embeddings:
                print(f"[warning] No trained model for city {city}, skipping.")
                continue

            # -------------------------------
            # Data from training
            # -------------------------------
            # emb_trained = self.city_perf_embeddings[city]
            # struct_scaler = self.city_struct_scaler[city]
            # struct_tree = self.city_struct_tree[city]

            df_train_idx = self.city_index_map[city]
            df_train_city = self.df.loc[df_train_idx]  # original training df for this city

            # Pre-extract training matrices
            perfX_train = df_train_city[self.perf_features].fillna(0).to_numpy()
            structX_train = df_train_city[self.structural_features].fillna(0).to_numpy()
            coords_train = df_train_city[["latitude", "longitude"]].to_numpy()

            # New listing data
            perfX_new = df_city[self.perf_features].fillna(0).to_numpy()
            structX_new = df_city[self.structural_features].fillna(0).to_numpy()
            coords_new = df_city[["latitude", "longitude"]].to_numpy()

            # -------------------------------
            # Distance matrix (new → train)
            # -------------------------------
            geo_km = self._haversine_km(coords_new, coords_train)  # shape (n_new, n_train)

            band_emb_list = []

            # -------------------------------
            # Construct band embeddings
            # -------------------------------
            for band_km in DISTANCE_BANDS:

                band_emb = np.zeros(
                    (len(df_city), len(self.perf_features) + len(self.structural_features))
                )

                for i in range(len(df_city)):
                    # neighbors in THIS band
                    mask = geo_km[i] <= band_km
                    neigh_idx = np.where(mask)[0]

                    if len(neigh_idx) == 0:
                        continue

                    # distance weights
                    w = 1 / (geo_km[i, neigh_idx] + 1e-6)
                    w = w / w.sum()

                    perf_agg = np.average(perfX_train[neigh_idx], axis=0, weights=w)
                    struct_agg = np.average(structX_train[neigh_idx], axis=0, weights=w)

                    # concat perf + struct
                    band_emb[i, : len(perf_agg)] = perf_agg
                    band_emb[i, len(perf_agg) :] = struct_agg

                band_emb_list.append(band_emb)

            # -------------------------------
            # Concatenate all bands
            # -------------------------------
            emb_new = np.concatenate(band_emb_list, axis=1)

            # Normalization to self.dim
            if emb_new.shape[1] > self.dim:
                # Use PCA fitted in training
                pca = self.city_pca[city]  # <-- needs to be saved during training
                emb_new = pca.transform(emb_new)
            elif emb_new.shape[1] < self.dim:
                pad = self.dim - emb_new.shape[1]
                emb_new = np.pad(emb_new, ((0, 0), (0, pad)), mode="constant")

            # -------------------------------
            # Wrap into DataFrame
            # -------------------------------
            df_emb = pd.DataFrame(
                emb_new, index=df_city.index, columns=[f"perf_emb_{i}" for i in range(self.dim)]
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
