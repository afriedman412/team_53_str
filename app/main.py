# app/main.py
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from app.api import inputs, predict, distances, debug, outputs, api
from app.core.loader import load_store
from app.core.deps import set_store, get_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # if a previous store is still alive in memory, skip
        get_store()
        print("🧠 Reusing existing DataStore (skipping reload)")
    except RuntimeError:
        print("🚀 App starting up — loading data...")
        store = load_store()
        set_store(store)
    yield
    print("🧹 App shutting down — cleanup complete.")

app = FastAPI(lifespan=lifespan, title="Chicago STR Prediction API")
app.include_router(predict.router)
app.include_router(distances.router)
app.include_router(inputs.router)
app.include_router(outputs.router)
app.include_router(debug.router)
app.include_router(api.router)

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/manual", response_class=HTMLResponse)
async def manual_form(request: Request):
    return templates.TemplateResponse("manual.html", {"request": request})


@app.post("/predict_manual")
async def predict_manual(request: Request):
    form = await request.form()
    # TODO: process features and return result
    return {"received": dict(form)}
