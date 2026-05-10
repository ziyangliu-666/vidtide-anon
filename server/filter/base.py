from abc import ABC, abstractmethod

class BaseFilter(ABC):
    name: str

    @abstractmethod
    def filter(self, candidates: list[dict], config: dict) -> list[dict]:
        ...

    def stats(self) -> dict:
        return {}
