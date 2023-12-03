import os
from typing import Optional

import requests
import streamlit as st

from skuld.ui.service import ImageService

CUSTOM_MASK_OPTION = "Свое изображение"


class Application:
    def __init__(self) -> None:
        self._image_service = ImageService()

        self._api_health_endpoint = os.environ["SKULD_UI_API_HEALTH_ENDPOINT"]
        self._api_predict_endpoint = os.environ["SKULD_UI_API_PREDICT_ENDPOINT"]

        self._mask_names = self._image_service.list_mask_names()
        self._mask_name = None
        self._mask: Optional[bytes] = None
        self._custom_mask = None
        self._angle_of_attack = None
        self._calculate = False

    def render(self) -> None:
        st.set_page_config(page_title="Нейросетевой CFD решатель", layout="centered", initial_sidebar_state="expanded")
        st.title("Нейросетевой CFD решатель")
        st.markdown("Описание проекта")

        with st.sidebar:
            self._render_sidebar()

        if self._calculate:
            self._render_result()

    def _render_sidebar(self) -> None:
        st.title("Конфигурация")
        api_available = self._api_available()

        if not api_available:
            st.warning("API недоступен. Но вы можете пользоваться кэшированными результатами.")
        else:
            self._mask_names.append(CUSTOM_MASK_OPTION)

        self._mask_name = st.selectbox(label="Расчетная область", options=self._mask_names)

        if self._mask_name == CUSTOM_MASK_OPTION:
            st.info("Загрузите свое изображение квадратной расчетной области. Препятствия должны быть черными, а зона потока - белой.")
            self._custom_mask = st.file_uploader(label="Загрузите изображение", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

        self._angle_of_attack = st.number_input(label="Угол атаки в градусах", min_value=-30, max_value=30, value=0, step=5)
        st.selectbox(label="Нейросеть", options=["Nidhoggr Medium"])

        self._mask = self._load_mask()

        if self._mask is not None:
            st.image(self._mask, use_column_width=True)

        self._calculate = st.button(label="Расчет")

    def _render_result(self) -> None:
        if self._custom_mask is not None:
            response = requests.post(
                self._api_predict_endpoint, params={"angle_of_attack": self._angle_of_attack}, files={"mask": self._mask}
            )

            if response.status_code == 200:
                st.image(response.content, use_column_width=True)
            else:
                st.warning(f"Расчет {response.text}")
        elif self._mask is None:
            st.info("Сначала необходимо выбрать изображение расчетной области.")
        else:
            self._image_service.get_result(mask_name=self._mask_name, angle_of_attack=self._angle_of_attack)

    def _api_available(self) -> bool:
        try:
            return requests.get(self._api_health_endpoint).status_code == 200
        except (requests.ConnectionError, requests.ConnectTimeout):
            return False

    def _load_mask(self) -> Optional[bytes]:
        if self._mask_name == CUSTOM_MASK_OPTION and self._custom_mask is not None:
            return self._custom_mask.getvalue()
        if self._mask_name != CUSTOM_MASK_OPTION:
            return self._image_service.get_mask_by_name(name=self._mask_name)
        return None


if __name__ == "__main__":
    Application().render()
