import pandas as pd
from app.utils.input_builder import assemble_prop_data
from app.model_api.pipeline_adapter import predict_from_saved
from app.schemas.property import AddressData
from typing import Dict


def process_address(address: AddressData) -> Dict:
    print("*** ADDRESS IN:", address.address)
    property_data = assemble_prop_data(address)
    print("*** TRAIN DISTANCE CHECK (dataclass):", property_data.dist_to_train_km)
    prop_dict = property_data.model_dump(exclude_none=False)
    print("*** TRAIN DISTANCE CHECK (post dump):", property_data.dist_to_train_km)
    prop_df = pd.DataFrame(prop_dict, index=[0])
    model_preds = predict_from_saved(prop_df)
    prop_dict.update(model_preds)
    return prop_dict
