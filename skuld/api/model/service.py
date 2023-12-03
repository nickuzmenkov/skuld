from skuld.api.model.adapter import ModelAdapter


class ModelService:
    def __init__(self) -> None:
        self._model_adapter = ModelAdapter()

    def predict(self, mask: bytes, angle_of_attack: float) -> bytes:
        return self._model_adapter(image=mask, angle_of_attack=angle_of_attack)
