# app/main.py
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, Request
import uvicorn
from contextlib import asynccontextmanager
from app.api import inputs, distances, debug, outputs, api, scenario_tests
from app.core.store import DataStore
from app.core.registry import set_store
from app.core.loader import load_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 App starting up — loading data...")
    store = load_store()
    set_store(store)
    app.state.store = store
    yield
    print("🧹 App shutting down — cleanup complete.")


app = FastAPI(lifespan=lifespan, title="Chicago STR Prediction API")


def get_store(request: Request) -> DataStore:
    return request.app.state.store


app.include_router(distances.router)
app.include_router(inputs.router)
app.include_router(outputs.router)
app.include_router(debug.router)
app.include_router(api.router)
app.include_router(scenario_tests.router)

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


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
