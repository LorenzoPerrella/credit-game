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
    #: Which continuous covariates are read as **bands** rather than as a line through their
    #: midpoints, and the reference band of each. Rule 12 of `docs/rules.md`.
    #:
    #: Carried by the specification rather than looked up, because it is part of what the model
    #: *is*: two specifications over the same covariates, one banded and one not, are different
    #: models with different parameter counts. Empty by default, so a caller that has not said
    #: gets the linear form -- which is what every test fixture wants, and what the selection
    #: did until `run_selection` started saying otherwise.
    banded: tuple[tuple[str, float], ...] = ()

    @property
    def covariates(self) -> list[str]:
        return [*self.continuous, *(name for name, _ in self.categorical)]

    @property
    def formula(self) -> str:
        """The formula these covariates make, with rule 12's bands read as bands.

        A banded covariate enters as a **factor**, one coefficient a band, against the band
        carrying the most loan-months. Until 18 October 2026 the selection did not apply that:
        steps 5 to 10 fitted `credit_score` as a straight line through its band midpoints while
        the published model read it as five levels.

        Rule 12 said the re-selection was **owed** -- "a covariate that survived on a straight
        line might not on a bend, or the reverse" -- and the cost of leaving it owed turned out
        to be more than a specification chosen on the wrong form. The two formulas differed, so
        a report could not start from the selection's own fit: 36 parameters handed to a model
        with 43 opened **5.1e+03 standard errors** out, the polish gave up, and a fit worth
        twenty minutes warm took nine and a half hours cold.
        """
        bands = dict(self.banded)
        terms = [
            f"C({name}, Treatment({bands[name]}))" if name in bands else name
            for name in self.continuous
        ]
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
