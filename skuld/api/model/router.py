from fastapi import APIRouter, Depends, Response, UploadFile

from skuld.api.model.service import ModelService

model_router = APIRouter(prefix="/model")


@model_router.post("/predict")
async def predict(mask: UploadFile, angle_of_attack: float, service: ModelService = Depends(ModelService)) -> Response:
    mask = await mask.read()
    content = service.predict(mask=mask, angle_of_attack=angle_of_attack)
    return Response(content=content, media_type="image/png")
