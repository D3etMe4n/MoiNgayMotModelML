# main.py
import uvicorn
from fastapi import FastAPI
from apis.routes import setup_routes
from logging_setup import setup_logging

app = FastAPI()

# Bật logging
setup_logging()

# Setup các route từ file routes.py
setup_routes(app)

if __name__ == "__main__":
    # uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    pass