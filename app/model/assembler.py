import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from app.core.col_control import PERF_FEATS, STRUCTURAL_FEATS
from app.core.model_config import MODEL_DATA, EMBEDDING_CONFIG
from app.model.embedder import PerformanceGraphEmbedderV3
from app.model.base_model import LightGBMRegressorCV
from app.model.rev_modeler import RevenueModeler


class Pops:
    def __init__(
        self,
        performance_cols: List = PERF_FEATS,
        structural_cols: List = STRUCTURAL_FEATS,
        embedder_path: str = None,
        embeddings_path: str = None,
        city_col: str = "city",
    ):
        self.performance_cols = performance_cols
        self.structural_cols = structural_cols
        self.modeling_cols = None

        self.embeddings = None
        self.embedder = None
        self.city_col = city_col

        # --- Load saved embeddings (optional) ---
        if embeddings_path is not None:
            self.embeddings = pd.read_csv(embeddings_path, index_col=0)

        # --- Load saved embedder (optional) ---
        if embedder_path is not None:
            self.embedder = PerformanceGraphEmbedderV3.load(embedder_path)
        else:
            self.embedder = PerformanceGraphEmbedderV3()
        return

    @property
    def can_embed_new(self):
        return self.embedder is not None

    def train_embeddings(self, training_df):
        """
        Ensures self.embeddings exists.
        If already loaded from file, do nothing.
        If embedder exists but embeddings are missing → generate via fit_transform().
        If neither exists → train embedder + embeddings from scratch.
        """
        # Already loaded externally
        if self.embeddings is not None:
            print("Embeddings already trained!!!")
            return self.embeddings

        # If no embedder, build + train it
        if self.embedder is not None:
            self._build_embedder()
            self.embeddings = self.embedder.fit_transform(training_df, self.city_col)
        return self.embeddings

    def fit(self, training_df):
        if self.embeddings is None:
            print("*** GETTING EMBEDDINGS")
            self.train_embeddings(training_df)
        df_with_embeddings = pd.concat([training_df, self.embeddings], axis=1)
        print("*** PRICE MODEL")
        self.price_model = self._run_model("price", data=df_with_embeddings)
        print("*** OCCUPANCY MODEL")
        self.occ_model = self._run_model("occupancy", data=df_with_embeddings)

        rm = RevenueModeler()

        print("*** REVENUE MODEL")
        X_corr, y_corr, _ = rm.prepare_training_data(
            df_with_embeddings, self.modeling_cols, self.price_model, self.occ_model
        )

        rm.fit(X_corr, y_corr)
        self.rev_model = rm

        print("*** BUILDING SHAP TREES")
        self._build_shap_trees()
        return

    def predict(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Predict price, occupancy, and revenue for new data.
        Uses:
          - structural_features for embedding KNN
          - modeling_cols + perf_emb_* for the models
        """
        if any(
            m is None for m in [self.embedder, self.price_model, self.occ_model, self.rev_model]
        ):
            raise ValueError("Pipeline not fitted yet.")

        required_cols = ["latitude", "longitude"]
        missing = [c for c in required_cols if c not in df_input.columns]
        if missing:
            raise ValueError(f"Missing required columns for embedding: {missing}")

        # 1) Build embeddings for input data (virgin or not)
        if not self.can_embed_new:
            raise RuntimeError("Embedder not available; cannot embed new listings.")
        new_embeddings = self.embedder.transform(df_input, city_col=self.city_col)

        # 2) Build feature matrix
        df_input = df_input[self.structural_cols]
        X_base = df_input.join(new_embeddings)

        # 3) Price + occupancy predictions (original scale)
        price_pred = self.price_model.predict(X_base)
        occ_pred = self.occ_model.predict(X_base)

        rev_pred = self.rev_model.predict(X_base, price_pred, occ_pred)

        self.df_with_embeds = X_base
        return pd.DataFrame(
            {
                "price_pred": price_pred,
                "occ_pred": occ_pred,
                "rev_final_pred": rev_pred,
            }
        )

    def _build_embedder(self):
        self.embedder = PerformanceGraphEmbedderV3(**EMBEDDING_CONFIG)
        return

    def _run_model(self, params_key, data=None, evaluate=False):
        params = MODEL_DATA[params_key]["params"]
        if data is not None:
            X, y = data

        else:
            target = MODEL_DATA[params_key]["target"]
            X = self.df[self.modeling_cols]
            y = self.df[target]

        model = LightGBMRegressorCV(
            params=params,
            transform="log1p",
            n_splits=5,
        )
        model.fit(X, y)
        return model

    def _build_shap_trees(self):
        import shap

        # pick one fold as representative (or average later if you want)
        self.price_explainer = shap.TreeExplainer(self.price_model._fold_models[0])
        self.occ_explainer = shap.TreeExplainer(self.occ_model._fold_models[0])
        # guessing RevenueModeler wraps a tree model as .model_
        self.rev_explainer = shap.TreeExplainer(self.rev_model.model._fold_models[0])
        return

    def shap_price(self, df_input: pd.DataFrame):
        if not hasattr(self, "price_explainer") or self.price_explainer is None:
            raise RuntimeError("Price SHAP explainer not available.")
        # ensure embeddings are present
        df_inp_w_emb = df_input.join(self.embedder.transform(df_input, city_col=self.city_col))
        X = df_inp_w_emb[self.modeling_cols]
        return self.price_explainer(X)

    def shap_occ(self, df_input: pd.DataFrame):
        if not hasattr(self, "occ_explainer") or self.occ_explainer is None:
            raise RuntimeError("Occupancy SHAP explainer not available.")
        df_inp_w_emb = df_input.join(self.embedder.transform(df_input, city_col=self.city_col))
        X = df_inp_w_emb[self.modeling_cols]
        return self.occ_explainer(X)

    # Utilities
    def evaluate(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(rmse, mae, r2)

    def save(self, path: str, save_embeddings: bool = True):
        """
        Save the full Pops bundle into a directory.

        Includes:
        - price/occ/rev models
        - modeling columns / config
        - SHAP explainers (if present)
        - embedder (PerformanceGraphEmbedderV3)
        - training embeddings (optional)
        """

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # -------------------------------
        # 1. Save core Pops payload
        # -------------------------------
        payload = {
            "price_model": self.price_model,
            "occ_model": self.occ_model,
            "rev_model": self.rev_model,
            "modeling_cols": self.modeling_cols,
            "performance_cols": self.performance_cols,
            "structural_cols": self.structural_cols,
            "city_col": self.city_col,
            # SHAP explainers (may be None)
            "price_explainer": getattr(self, "price_explainer", None),
            "occ_explainer": getattr(self, "occ_explainer", None),
            "rev_explainer": getattr(self, "rev_explainer", None),
        }

        joblib.dump(payload, path / "pops.joblib")

        # -------------------------------
        # 2. Save embedder
        # -------------------------------
        if self.embedder is not None:
            joblib.dump(self.embedder, path / "embedder.joblib")

        # -------------------------------
        # 3. Save embeddings (optional)
        # -------------------------------
        if save_embeddings and getattr(self, "embeddings", None) is not None:
            emb = self.embeddings.astype("float64")
            emb.to_parquet(path / "embeddings.parquet")

        # -------------------------------
        # 4. Optional metadata file
        # -------------------------------
        metadata = {
            "version": "1.0",
            "notes": "Pops bundle with models + embedder",
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"[saved] model bundle at {path}")

    @classmethod
    def load(cls, path: str):
        """
        Load a Pops bundle directory created by Pops.save().
        Automatically loads:
        - Pops payload (models, explainers, config)
        - embedder
        - embeddings (if present)
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Pops.load: path does not exist: {path}")

        # -------------------------------
        # 1. Load Pops payload
        # -------------------------------
        payload = joblib.load(path / "pops.joblib")

        obj = cls(
            performance_cols=payload["performance_cols"],
            structural_cols=payload["structural_cols"],
            city_col=payload["city_col"],
        )

        obj.price_model = payload["price_model"]
        obj.occ_model = payload["occ_model"]
        obj.rev_model = payload["rev_model"]
        obj.modeling_cols = payload["modeling_cols"]

        # SHAP explainers
        obj.price_explainer = payload.get("price_explainer")
        obj.occ_explainer = payload.get("occ_explainer")
        obj.rev_explainer = payload.get("rev_explainer")

        # -------------------------------
        # 2. Load embedder
        # -------------------------------
        embedder_path = path / "embedder.joblib"
        if embedder_path.exists():
            obj.embedder = joblib.load(embedder_path)
        else:
            obj.embedder = None

        # -------------------------------
        # 3. Load precomputed embeddings
        # -------------------------------
        embeddings_path = path / "embeddings.parquet"
        if embeddings_path.exists():
            obj.embeddings = pd.read_parquet(embeddings_path)
        else:
            obj.embeddings = None

        print(f"[loaded] Pops bundle from {path}")
        return obj

    def summary(self) -> dict:
        return {
            "price_model": self.price_model is not None,
            "occ_model": self.occ_model is not None,
            "rev_model": self.rev_model is not None,
            "modeling_cols": len(self.modeling_cols) if self.modeling_cols is not None else 0,
            "embedder_loaded": self.embedder is not None,
            "precomputed_embeddings": self.embeddings is not None,
            "shap_price": hasattr(self, "price_explainer") and self.price_explainer is not None,
            "shap_occ": hasattr(self, "occ_explainer") and self.occ_explainer is not None,
            "shap_rev": hasattr(self, "rev_explainer") and self.rev_explainer is not None,
        }
