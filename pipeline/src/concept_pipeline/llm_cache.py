"""Record and replay of model calls for the pipeline, keyed by content (plan section 1.2).

Three modes: replay (the default; never sends, a missing answer is an error), record (sends only
what is missing) and fresh (always sends, to measure stability, and never overwrites a recorded
answer). Tokens are counted for every answer used. The sender is passed in, so this module
itself has no network code and tests run with a fake.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

# Short enough for a file name, long enough that two different calls never collide in practice.
KEY_HEX_CHARS = 24
FRESH_SUFFIX = ".fresh.jsonl"


class Mode(str, Enum):
    """How the cache treats a call."""

    REPLAY = "replay"
    RECORD = "record"
    FRESH = "fresh"


@dataclass(frozen=True)
class LlmCall:
    """Everything that decides a model's answer; any change gives a new cache key."""

    purpose: str  # e.g. "normalize"; part of the file name, so a folder is easy to read
    prompt_version: str
    model: str
    system: str
    prompt: str


@dataclass(frozen=True)
class LlmAnswer:
    """A model's reply text and what it cost."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    served_model: str = ""


class CacheMiss(LookupError):
    """Raised in replay mode when a call has no recorded answer."""


@dataclass
class Usage:
    """Running totals for one cache: answers used, how many were sent, and their tokens."""

    answers: int = 0
    sent: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, answer: LlmAnswer, was_sent: bool) -> None:
        """Count one answer that was used."""
        self.answers += 1
        self.sent += was_sent
        self.input_tokens += answer.input_tokens
        self.output_tokens += answer.output_tokens


def call_key(call: LlmCall) -> str:
    """Return the content hash that names a call's cache file."""
    identity = json.dumps(
        [call.prompt_version, call.model, call.system, call.prompt], ensure_ascii=False
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:KEY_HEX_CHARS]


@dataclass
class LlmCache:
    """A folder of recorded answers and the mode that decides when to send.

    Args:
        folder: Where answers are kept; created on the first write.
        mode: replay, record or fresh.
        send: Sends one call and returns the answer; needed only for record and fresh.
    """

    folder: Path
    mode: Mode = Mode.REPLAY
    send: Callable[[LlmCall], LlmAnswer] | None = None
    usage: Usage = field(default_factory=Usage)

    def path(self, call: LlmCall) -> Path:
        """Return the file that holds a call's recorded answer."""
        return self.folder / f"{call.purpose}-{call_key(call)}.json"

    def answer(self, call: LlmCall) -> LlmAnswer:
        """Return the answer to a call, from the cache or by sending it, as the mode allows.

        Raises:
            CacheMiss: In replay mode, when no answer is recorded.
            ValueError: When the mode must send and no sender was given.
        """
        recorded = self._read(call)
        if self.mode is Mode.REPLAY:
            if recorded is None:
                raise CacheMiss(f"no recorded answer for {self.path(call).name}")
            self.usage.add(recorded, was_sent=False)
            return recorded
        if self.mode is Mode.RECORD and recorded is not None:
            self.usage.add(recorded, was_sent=False)
            return recorded
        if self.send is None:
            raise ValueError(f"mode {self.mode.value} needs a sender")
        answer = self.send(call)
        self.usage.add(answer, was_sent=True)
        if recorded is None:
            self._write(call, answer)
        elif self.mode is Mode.FRESH:
            self._append_fresh(call, answer)
        return answer

    def _read(self, call: LlmCall) -> LlmAnswer | None:
        """Return the recorded answer, or None when there is none."""
        path = self.path(call)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return LlmAnswer(**data["answer"])

    def _write(self, call: LlmCall, answer: LlmAnswer) -> None:
        """Record an answer atomically, so an interrupted run never leaves half a file."""
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.path(call)
        temporary = path.with_suffix(".tmp")
        record = {"call": asdict(call), "answer": asdict(answer)}
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(temporary, path)

    def _append_fresh(self, call: LlmCall, answer: LlmAnswer) -> None:
        """Keep a fresh answer next to the recorded one, for the stability check."""
        path = self.path(call).with_suffix(FRESH_SUFFIX)
        with path.open("a", encoding="utf-8") as fresh:
            fresh.write(json.dumps(asdict(answer), ensure_ascii=False) + "\n")
