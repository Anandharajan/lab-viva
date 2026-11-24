"""A2: computerised adaptive viva: maximum Fisher information subject to concept coverage
and no repeats within a batch. Stop when SE(theta) < 0.35 or after 8 items."""
from __future__ import annotations

from dataclasses import dataclass, field

from .irt import eap, information


@dataclass
class Item:
    iid: str
    concept: str
    a: float
    b: float


@dataclass
class Session:
    student: str
    asked: list[str] = field(default_factory=list)
    responses: list[tuple[float, float, int]] = field(default_factory=list)
    covered: set[str] = field(default_factory=set)

    @property
    def theta(self) -> tuple[float, float]:
        return eap(self.responses)


def next_item(items: list[Item], s: Session, used_by_batch: set[str], se_stop: float = 0.35,
              max_items: int = 8) -> Item | None:
    th, se = s.theta
    if len(s.asked) >= max_items or (s.asked and se < se_stop):
        return None
    pool = [i for i in items if i.iid not in s.asked and i.iid not in used_by_batch]
    fresh = [i for i in pool if i.concept not in s.covered]
    cand = fresh or pool  # coverage constraint relaxes only when every concept is covered
    if not cand:
        return None
    return max(cand, key=lambda i: (information(th, i.a, i.b), i.iid))


def record(s: Session, item: Item, correct: bool) -> None:
    s.asked.append(item.iid)
    s.covered.add(item.concept)
    s.responses.append((item.a, item.b, int(correct)))
