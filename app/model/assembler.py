import pandas as pd
import numpy as np
import joblib
import json
import io
import base64
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from app.core.col_control import PERF_FEATS, STRUCTURAL_FEATS
from app.core.model_config import MODEL_DATA, EMBEDDING_CONFIG
from app.core.config import CSG_PALETTE
from app.model.embedder import PerformanceGraphEmbedderV3
from app.model.base_model import LightGBMRegressorCV
from app.model.rev_modeler import RevenueModeler
from app.model.helpers import get_modeling_columns


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
            self._pull_embeddings()
        else:
            self._build_embedder()
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
        """
        REMOVE ZEROS WHEN TRAINING
        df_ = df[df['estimated_revenue_l365d'] > 0]
        """
        if self.embeddings is None:
            print("*** GETTING EMBEDDINGS")
            self.train_embeddings(training_df)
        df_with_embeddings = pd.concat([training_df, self.embeddings], axis=1)
        print("*** PRICE MODEL")
        self.price_model = self._run_model("price", df_with_embeddings, evaluate=True)
        print("*** OCCUPANCY MODEL")
        self.occ_model = self._run_model("occupancy", df_with_embeddings, evaluate=True)

        rm = RevenueModeler()

        print("*** REVENUE MODEL")
        X_corr, y_corr, _ = rm.prepare_training_data(
            df_with_embeddings, self.price_model, self.occ_model
        )

        rm.fit(X_corr, y_corr)
        self.rev_model = rm

        print("*** BUILDING SHAP TREES")
        self._build_shap_trees()
        print("*** DONE TRAINING")
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
        print("*** GETTING EMBEDDINGS")
        if not self.can_embed_new:
            raise RuntimeError("Embedder not available; cannot embed new listings.")
        new_embeddings = self.embedder.transform(df_input, city_col=self.city_col)

        # 2) Build feature matrix
        df_input = df_input[self.structural_cols]
        X_base = df_input.join(new_embeddings)

        # 3) Price + occupancy predictions (original scale)
        print("*** PREDICTING PRICE")
        price_pred = self.price_model.predict(X_base)
        print("*** PREDICTING OCCUPANCY")
        occ_pred = self.occ_model.predict(X_base)
        print("*** PREDICTING REVENUE")
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

    def _pull_embeddings(self):
        if self.embedder is not None:
            if self.embedder.city_perf_embeddings is not None:
                self.embeddings = self.embedder.city_perf_embeddings
            else:
                print("Embedder has not been trained!")

        else:
            print("No embedder loaded!")
        return

    def _run_model(self, params_key, df, evaluate=False):
        params = MODEL_DATA[params_key]["params"]
        transform = MODEL_DATA[params_key]["transform"]
        target = MODEL_DATA[params_key]["target"]
        self.modeling_cols = get_modeling_columns(df)
        X = df[self.modeling_cols]
        y = df[target]

        model = LightGBMRegressorCV(
            params=params,
            transform=transform,
            n_splits=5,
        )

        model.fit(X, y)
        if evaluate:
            preds = model.predict(X)
            self.evaluate(y, preds)
        return model

    def _build_shap_trees(self):
        import shap

        # price + occupancy = simple
        self.price_explainer = shap.TreeExplainer(self.price_model._fold_models[0])
        self.occ_explainer = shap.TreeExplainer(self.occ_model._fold_models[0])

        # revenue = complicated
        # rev_model.model is a LightGBMRegressorCV
        # so use its underlying _fold_models[0]
        base_model = self.rev_model.model._fold_models[0]

        # SHAP on the correction model needs X_corr columns, not X_base
        self.rev_X_cols = self.rev_model.X_corr_cols  # you must store this during fit
        self.rev_explainer = shap.TreeExplainer(base_model)
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

    def shap_rev(self, df_input):
        """
        Because it's more complicated.
        """
        if len(df_input.filter(regex="perf")) == 0:
            df_input = df_input.join(self.embedder.transform(df_input, city_col=self.city_col))
        price_pred = self.price_model.predict(df_input)
        occ_pred = self.occ_model.predict(df_input)

        rev_base = np.clip(price_pred * occ_pred, 1e-6, None)
        rev_base_log = np.log1p(rev_base)

        X_corr = df_input.copy()
        X_corr["rev_base_log"] = rev_base_log
        X_corr = X_corr[self.rev_model.X_corr_cols]

        shap_values_rev = self.rev_explainer.shap_values(X_corr)
        return shap_values_rev

    # Utilities
    def evaluate(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(rmse, mae, r2)

    def predict_and_explain(self, permo_df, amenities=None, return_plot=True):
        """
        Full revenue explanation pipeline for permutation or real listings.

        Produces:
        - X_corr (correct revenue correction feature matrix)
        - SHAP values for the revenue model
        - Global feature importance bar plot
        - Amenity uplift table + bar chart
        - Returns a dict with all components
        """

        # -----------------------
        # 1. Predict first to build df_with_embeds correctly
        # -----------------------
        preds = self.predict(permo_df)  # populates self.df_with_embeds

        df_base = self.df_with_embeds.copy()

        # -----------------------
        # 2. Build X_corr EXACTLY as in training
        # -----------------------
        price_pred = preds["price_pred"]
        occ_pred = preds["occ_pred"]

        rev_base_log = np.log1p(price_pred * occ_pred)

        X_corr = df_base.copy()
        X_corr["rev_base_log"] = rev_base_log

        # Keep features in correct training order
        X_corr = X_corr[self.rev_model.X_corr_cols]

        # -----------------------
        # 3. SHAP values
        # -----------------------
        shap_vals = self.rev_explainer.shap_values(X_corr)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        # -----------------------
        # 4. Amenity uplift calculations
        # -----------------------
        uplifts = {}
        if amenities is None:  # Auto-detect binary amenities the model knows about
            amenities = [
                c
                for c in permo_df.columns
                if c
                in [
                    "beds",
                    "pool",
                    "hot_tub",
                    "gym",
                    "accommodates",
                    "housekeeping",
                    "free_parking",
                ]
            ]

        # structural features you want to hold fixed
        control_cols = ["accommodates"]

        for feat in amenities:
            diffs = []

            # iterate over all combos of accommodations × beds
            for acc, sub in permo_df.groupby(control_cols):

                # require both amenity == 0 and amenity == 1 to exist
                if sub[feat].nunique() < 2:
                    continue

                pred_sub = preds.loc[sub.index]

                rev1 = pred_sub.loc[sub[feat] == 1, "rev_final_pred"].mean()
                rev0 = pred_sub.loc[sub[feat] == 0, "rev_final_pred"].mean()

                diffs.append(rev1 - rev0)

            # uplift = average across all controlled diffs
            uplifts[feat] = np.mean(diffs) if diffs else np.nan

        uplift_series = pd.Series(uplifts).sort_values()

        # -----------------------
        # 5. Plots
        # -----------------------
        if return_plot:

            fig, ax = plt.subplots(figsize=(6, 3))
            uplift_series.sort_values().plot(
                kind="barh",
                ax=ax,
                color=CSG_PALETTE[0],  # gold
            )

            ax.set_title("Amenity Uplift (Annual Revenue)", color=CSG_PALETTE[1])
            ax.set_xlabel("Revenue change ($)")
            ax.axvline(0, color=CSG_PALETTE[5], linewidth=1)  # near black
            ax.grid(axis="x", alpha=0.15)
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            png_b64 = base64.b64encode(buf.read()).decode("ascii")

        # -----------------------
        # 6. Return everything for downstream analysis
        # -----------------------
        return {
            "shap_values": shap_vals,
            "X_corr": X_corr,
            "uplift_table": uplift_series,
            "uplift_char_png": png_b64,
            "preds": preds,
        }

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
