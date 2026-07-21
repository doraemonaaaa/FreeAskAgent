"""Small dependency-free RAG store for navigation guidance and experience."""

import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


class NavigationRAG:
    """Retrieve relevant task hints using deterministic lexical similarity.

    The store intentionally has no embedding-model dependency, so it can be used
    alongside two large VLMs without consuming additional GPU memory.
    """

    def __init__(self, documents: Optional[Iterable[str]] = None, path: Optional[str] = None):
        self.path = Path(path).expanduser() if path else None
        self.documents: List[str] = []
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.documents.extend(data if isinstance(data, list) else data["documents"])
        if documents:
            self.add(documents)

    def add(self, documents: Iterable[str]) -> None:
        for document in documents:
            document = document.strip()
            if document and document not in self.documents:
                self.documents.append(document)

    def search(self, query: str, top_k: int = 3) -> Sequence[str]:
        query_tokens = set(self._tokens(query))
        if not query_tokens:
            return ()
        ranked = []
        for index, document in enumerate(self.documents):
            tokens = set(self._tokens(document))
            overlap = len(query_tokens & tokens)
            if overlap:
                ranked.append((overlap / max(len(query_tokens | tokens), 1), index, document))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return tuple(item[2] for item in ranked[:top_k])

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.documents, ensure_ascii=True, indent=2), encoding="utf-8")

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())
