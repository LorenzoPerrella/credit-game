"""What a model is, as a value.

A specification is the covariates and the reference levels, and the formula is derived from
them rather than stored beside them -- so a specification and its formula cannot disagree.
`plus` and `minus` return new ones, because backward elimination is a sequence of models and
not a model being edited: every step of a selection has a specification that still exists
afterwards, which is what makes the record readable and the floor computable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Specification:
    """The covariates a model reads, and the formula that says so."""

    continuous: tuple[str, ...]
    categorical: tuple[tuple[str, str], ...] = ()

    @property
    def covariates(self) -> list[str]:
        return [*self.continuous, *(name for name, _ in self.categorical)]

    @property
    def formula(self) -> str:
        terms = [*self.continuous]
        terms += [f"C({name}, Treatment('{level}'))" for name, level in self.categorical]
        return " + ".join(terms)

    def plus(self, name: str, *, reference: str | None = None) -> Specification:
        if reference is not None:
            return replace(self, categorical=(*self.categorical, (name, reference)))
        return replace(self, continuous=(*self.continuous, name))

    def minus(self, name: str) -> Specification:
        return replace(
            self,
            continuous=tuple(term for term in self.continuous if term != name),
            categorical=tuple(pair for pair in self.categorical if pair[0] != name),
        )
