from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from app.services.openai_compatible_client import OpenAICompatibleClient


class ConnectionWorker(QObject):
    result_signal = Signal(bool, str)
    finished_signal = Signal()

    def __init__(self, client: OpenAICompatibleClient, model_name: str) -> None:
        super().__init__()
        self.client = client
        self.model_name = model_name

    @Slot()
    def run(self) -> None:
        try:
            ok, message = self.client.test_connection(self.model_name)
            self.result_signal.emit(ok, message)
        finally:
            self.finished_signal.emit()
