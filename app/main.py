# app/main.py
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, Request
import uvicorn
from contextlib import asynccontextmanager
from app.api import inputs, distances, debug, outputs, api
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

templates = Jinja2Templates(directory="templates")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
