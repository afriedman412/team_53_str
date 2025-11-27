import sys
from app.model import base_model as _base_model
from app.model import assembler as _assembler
from app.model import embedder as _embedder
from app.model import rev_modeler as _rev_modeler
from app.model import helpers as _helpers

# backwards compatibility for old pickle paths
sys.modules["base_model"] = _base_model
sys.modules["assembler"] = _assembler
sys.modules["embedder"] = _embedder
sys.modules["rev_modeler"] = _rev_modeler
sys.modules["helpers"] = _helpers
