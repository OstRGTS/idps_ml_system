from prevention.api import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run("prevention.api:app", host="0.0.0.0", port=8000, reload=True)