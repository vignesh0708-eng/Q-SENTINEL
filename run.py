"""Start the server:  python run.py   (env: HOST, PORT)"""
import os
from app.main import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)), debug=False)
