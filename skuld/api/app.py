from fastapi import FastAPI, Response

from skuld.api.model.router import model_router

app = FastAPI()
app.include_router(model_router)


@app.get("/health")
async def health() -> Response:
    return Response()
