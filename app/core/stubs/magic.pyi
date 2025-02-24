from typing import Optional


class Magic:
    def __init__(
        self,
        mime: bool = False,
        magic_file: Optional[str] = None
    ) -> None: ...
    def from_buffer(self, buffer: bytes) -> str: ...
    def from_file(self, filename: str) -> str: ... 