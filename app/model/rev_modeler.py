import numpy as np
from app.model.base_model import LightGBMRegressorCV
from app.core.model_config import MODEL_DATA


class RevenueModeler:
    """
    Handles revenue correction modeling:
      - builds rev_corr_target
      - fits residual model
      - predicts final revenue
    """

    def __init__(self, params=None):
        self.model = None
        if params is None:
            self.params = MODEL_DATA["revenue_corr"]["params"]

    # -------------------------------------------------------------
    def prepare_training_data(self, df, modeling_cols, price_model, occ_model):
        """
        Prepares:
         - rev_true_log
         - rev_base_log
         - rev_corr_target
        """
        X = df[modeling_cols]

        # base predictions
        price_pred = price_model.predict(X)
        occ_pred = occ_model.predict(X)

        rev_true = df["estimated_revenue_l365d"].clip(lower=1e-6)
        rev_base = (price_pred * occ_pred).clip(1e-6)

        # log transforms
        rev_true_log = np.log1p(rev_true)
        rev_base_log = np.log1p(rev_base)

        # residual
        rev_corr_target = rev_true_log - rev_base_log

        # features for correction model
        X_corr = X.copy()
        X_corr["rev_base_log"] = rev_base_log

        return X_corr, rev_corr_target, rev_base_log

    # -------------------------------------------------------------
    def fit(self, X_corr, y_corr):
        self.model = LightGBMRegressorCV(params=self.params, transform="none")
        self.model.fit(X_corr, y_corr)
        return self

    # -------------------------------------------------------------
    def predict(self, X_base, price_pred, occ_pred):
        """
        Predict final revenue:
          - compute base_log from price * occ
          - run residual model
          - expm1(base_log + residual)
        """
        rev_base = (price_pred * occ_pred).clip(1e-6)
        base_log = np.log1p(rev_base)

        X = X_base.copy()
        X["rev_base_log"] = base_log

        corr_pred = self.model.predict(X)

        rev_log_final = base_log + corr_pred
        return np.expm1(rev_log_final)
