# logging_setup.py
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),           # In ra console
            logging.FileHandler("app.log")     # Ghi log vào file
        ]
    )