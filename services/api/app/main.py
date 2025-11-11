from __future__ import annotations

from fastapi import FastAPI

from .config import get_settings
from .db.session import init_db
from .routers import predictions


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="HR Job Change Prediction API")

    @app.on_event("startup")
    def on_startup() -> None:
        init_db()

    app.include_router(predictions.router)
    return app


app = create_app()
