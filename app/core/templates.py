from fastapi.templating import Jinja2Templates
from app.utils import filters

templates = Jinja2Templates(directory="templates")

# register filters globally
templates.env.filters["currency"] = filters.currency
templates.env.filters["comma"] = filters.comma
templates.env.filters["percent"] = filters.percent
