from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from app.services.openai_compatible_client import OpenAICompatibleClient


class StreamWorker(QObject):
    chunk_received = Signal(str)
    started_signal = Signal()
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(
        self,
        client: OpenAICompatibleClient,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        use_stream: bool,
    ) -> None:
        super().__init__()
        self.client = client
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_stream = use_stream

    @Slot()
    def run(self) -> None:
        self.started_signal.emit()
        try:
            if self.use_stream:
                self.client.stream_chat(
                    model_name=self.model_name,
                    system_prompt=self.system_prompt,
                    user_prompt=self.user_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    on_delta=self.chunk_received.emit,
                )
            else:
                result = self.client.non_stream_chat(
                    model_name=self.model_name,
                    system_prompt=self.system_prompt,
                    user_prompt=self.user_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                self.chunk_received.emit(result)
        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            self.finished_signal.emit()
