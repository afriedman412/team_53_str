import pandas as pd
from app.utils.input_builder import build_input
from app.models.pipeline_adapter import predict_from_saved


def process_address(address):
    property_data = build_input(address)
    prop_dict = property_data.model_dump(exclude_none=False)
    prop_df = pd.DataFrame(prop_dict, index=[0])
    model_preds = predict_from_saved(prop_df)
    prop_dict.update(model_preds)
    return prop_dict
