# app/main.py
from dotenv import load_dotenv
import os
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request
import uvicorn
from contextlib import asynccontextmanager
from app.api import inputs, distances, debug, api, outputs
from app.core.store import DataStore
from app.core.registry import set_store
from app.core.loader import load_store

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Skip heavy startup during tests
    if os.environ.get("TESTING"):
        print("⚠️ Skipping lifespan (test mode)")
        yield
        return

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

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/ai_test", response_class=HTMLResponse)
async def ai_test(request: Request):
    return templates.TemplateResponse("ai_test.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
