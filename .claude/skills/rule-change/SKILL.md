---
name: rule-change
description: Declare, amend or retire a modelling rule, acceptance criterion or model form in credit-game, and record a prediction or a result in the rules register. Use before any fit whose outcome a rule judges, when a result falsifies a prediction, and whenever docs/rules.md or docs/rules_register.md is edited.
---

# Changing a rule

`docs/rules.md` holds the rules; `docs/rules_register.md` is the dated, append-only record of
what changed, what was predicted and what was measured. A rule that judges a fit can be trusted
only if it was written before the fit. This skill is the procedure that keeps that true.

## Before a fit a rule will judge

1. **Declare the form and the criterion in `docs/rules.md`**, with the measurement that motivates
   them. Every constant the form needs (knots, references, bands) is computed from the
   **estimation table**, never from the test window. Write down which table, so that a tie cannot
   be settled by convenience. The age reference band was declared on a 0.9% margin for exactly
   that reason.
2. **Write the prediction in the register before the run**: which numbers will move, in which
   direction, what would falsify it, and what is declared *not* to be evidence. A better likelihood
   is never evidence, because rule 2 exists to refuse it. An in-sample reading of a factor on the
   criterion's own grouping is not evidence either: a saturated factor reproduces its groups
   exactly, so it reads 1.000 by construction.
3. **Read the test window once.** The anchoring window (2022-01..2024-12) sets the level and
   nothing else. The test window is the months after it.

## After the run

- **Publish the outcome whatever it is**, against the prediction, in the register: the falsified
  ones especially. Recording the predictions is what made the falsified ones knowable.
- **A number is not interpretable without the model *and* the window it is of.** Name the fit's
  hash, its `sample` and the window beside every figure. One coefficient table was quoted for a
  month from the "after 2010-06" stability sub-sample instead of the model.
- **Verify a claim against the code that computes it, not against the algebra.** A fit was refused
  on an algebraic argument about `H(t | fixed eta)` when `episode_hazards` takes its increment at
  fixed covariates *within the month*. One call to that function would have shown the hazard
  positive at every age.

## Amending a rule after a result

An amendment written after seeing a result has the shape of tuning, so it is admissible only if:

- **every input to its argument predates the result** (the model's own definitions, earlier
  register entries, committed observed counts), and
- **it cannot turn a failure into a pass**: no acceptance criterion moves.

It also needs a second signatory, the user, and a register entry. If a bullet of a prediction
turns out to be ill-defined, withdraw it **in the register, dated, saying what had already been
seen** when that was found. Never edit the original prediction text.

**Prefer the honest route even when an amendment would arrive at the same place.** The band form
was refused on a misreading and an amendment was drafted to license the fallback. The out-of-sample
measurement licensed it anyway, one run later, under the rule's own sentence. The amendment was
wrong and also unnecessary.

## Withdrawals and corrections

Keep the withdrawn text under a notice rather than deleting it, with the date and what replaced
it. A withdrawn criterion that stops being reported is one nobody can check the withdrawal of.
