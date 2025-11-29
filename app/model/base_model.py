import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def _sanitize_params(params):
    clean = params.copy()
    if clean.get("objective") is None:
        clean["objective"] = "regression"
    return clean


def _identity(x):
    return x


class LightGBMRegressorCV(BaseEstimator, RegressorMixin):
    """
    sklearn-compatible LightGBM CV regressor.
    - Supports transform='none' or 'log1p'
    - CV training with early stopping
    - OOF predictions, feature importances
    - Fully compatible with GridSearchCV / RandomizedSearchCV
    """

    def __init__(
        self,
        params=None,
        n_splits=5,
        random_state=42,
        transform="none",
        clip_pred_lower=None,
    ):
        # sklearn requires ALL constructor args to be stored as attributes
        self.params = params if params is not None else {}
        self.n_splits = n_splits
        self.random_state = random_state
        self.transform = transform
        self.clip_pred_lower = clip_pred_lower

        # outputs
        self._fold_models = []
        self.feature_importance_ = None
        self.oof_pred_ = None

    # -------------------------------------------------------------
    # Required by sklearn
    # -------------------------------------------------------------
    def get_params(self, deep=True):
        return {
            "params": self.params,
            "n_splits": self.n_splits,
            "random_state": self.random_state,
            "transform": self.transform,
            "clip_pred_lower": self.clip_pred_lower,
        }

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    # -------------------------------------------------------------
    # Target transform helpers
    # -------------------------------------------------------------
    def _setup_transform(self):
        if self.transform == "none":
            self._forward = _identity
            self._inverse = _identity
        elif self.transform == "log1p":
            self._forward = np.log1p
            self._inverse = np.expm1
        else:
            raise ValueError("transform must be 'none' or 'log1p'")

    # -------------------------------------------------------------
    # FIT
    # -------------------------------------------------------------
    def fit(self, X, y):
        """
        sklearn fit(X,y) → trains CV models.
        Stores:
        - self._fold_models
        - self.oof_pred_
        - self.feature_importance_
        """
        self.X_ = X.copy()
        self.y_ = y.copy()

        self._setup_transform()

        Xc = self.X_.copy()
        for c in Xc.select_dtypes("object"):
            Xc[c] = Xc[c].astype("category")
        cat_cols = Xc.select_dtypes("category").columns.tolist()

        y_trans = self._forward(self.y_)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        oof_pred = np.zeros(len(Xc))
        oof_pred_trans = np.zeros(len(Xc))
        feat_imp_all = []

        self._fold_models = []

        for fold, (tr_idx, val_idx) in enumerate(kf.split(Xc), 1):
            print(f"Fold {fold}")

            X_tr, X_val = Xc.iloc[tr_idx], Xc.iloc[val_idx]
            y_tr, y_val = y_trans.iloc[tr_idx], y_trans.iloc[val_idx]

            dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
            dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols)

            booster = lgb.train(
                _sanitize_params(self.params),
                dtr,
                num_boost_round=5000,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(500),
                ],
            )
            self._fold_models.append(booster)

            # Predict
            val_pred_trans = booster.predict(X_val, num_iteration=booster.best_iteration)
            val_pred = self._inverse(val_pred_trans)
            if self.clip_pred_lower is not None:
                val_pred = np.clip(val_pred, self.clip_pred_lower, None)

            oof_pred[val_idx] = val_pred
            oof_pred_trans[val_idx] = val_pred_trans

            fold_imp = pd.DataFrame(
                {
                    "feature": booster.feature_name(),
                    "importance": booster.feature_importance("gain"),
                    "fold": fold,
                }
            )
            feat_imp_all.append(fold_imp)

        self.oof_pred_ = oof_pred
        self.feature_importance_ = pd.concat(feat_imp_all, axis=0).reset_index(drop=True)

        return self

    # -------------------------------------------------------------
    # PREDICT
    # -------------------------------------------------------------
    def predict(self, X):
        """
        sklearn predict(X)
        Averages predictions across CV fold models.
        """
        if not self._fold_models:
            raise ValueError("Must call fit() first.")

        Xp = X.copy()
        for c in Xp.select_dtypes("object"):
            Xp[c] = Xp[c].astype("category")

        preds_trans = [
            model.predict(Xp, num_iteration=model.best_iteration) for model in self._fold_models
        ]

        mean_trans = np.mean(preds_trans, axis=0)
        preds = self._inverse(mean_trans)

        if self.clip_pred_lower is not None:
            preds = np.clip(preds, self.clip_pred_lower, None)

        return preds

    # -------------------------------------------------------------
    # Extra helpers
    # -------------------------------------------------------------
    def score(self, X, y):
        """sklearn-style R²."""
        return r2_score(y, self.predict(X))

    def evaluate_oof(self):
        """Convenience metrics on OOF predictions."""
        rmse = np.sqrt(mean_squared_error(self.y_, self.oof_pred_))
        mae = mean_absolute_error(self.y_, self.oof_pred_)
        r2 = r2_score(self.y_, self.oof_pred_)
        print(f"OOF → RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        return {"rmse": rmse, "mae": mae, "r2": r2}

    def plot_importance(self, top_n=20):
        imp = (
            self.feature_importance_.groupby("feature")["importance"]
            .mean()
            .sort_values(ascending=False)
        )
        imp.head(top_n).plot(kind="barh", figsize=(8, 6))
        plt.gca().invert_yaxis()
        plt.title("Feature Importance (mean gain)")
        plt.show()
        return imp
