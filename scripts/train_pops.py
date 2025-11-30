from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent  # WORKDIR
APP_DIR = ROOT_DIR / "app"
sys.path.append(str(ROOT_DIR))


def pops_trainer():
    from app.model.assembler import Pops
    from app.core.model_config import DF_PATH
    from app.utils.perm_builder import SCENARIO_DEFAULTS
    import pandas as pd
    import json

    df = pd.read_csv(DF_PATH)
    df_ = df[df["estimated_revenue_l365d"] > 0]

    pops = Pops()
    pops.fit(df_)

    json_path = "data/pops_training_fodder.json"
    with open(json_path) as f:
        jjj = f.read()
        base = json.loads(jjj)

    base.update(SCENARIO_DEFAULTS)

    preds = pops.predict(base)
    print(preds)

    pops.save("new_pops")
    return


pops_trainer()
