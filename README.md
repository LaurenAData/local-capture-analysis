# Local capture and leakage in event-driven urban spending

A sensitivity analysis of how a visitor dollar moves through a host-city economy,
modeled as an absorbing Markov chain.

## The question

Cities that host large sporting events publish detailed economic impact projections
beforehand and almost never verify them afterward. The 2026 FIFA World Cup was hosted
across sixteen North American cities with projections in the tens of billions, and
several host cities created supplier programs specifically to route procurement toward
local and diverse businesses. Whether those programs changed anything is largely unknown.

Part of the problem is that impact is reported as a total, and a total cannot distinguish
a dollar that circulates six times in a neighborhood from one that leaves the region on
its first transaction. That distinction is a question about network structure.

## What this does

Models an entering dollar as an absorbing Markov chain over business and institutional
nodes, with a single absorbing state representing exit from the regional economy, and
computes the expected number of local transactions generated before exit.

**The transition probabilities are not known and are not asserted.** No public dataset
gives the share of stadium-district spending captured by independent businesses. The
parameters are swept across plausible ranges so that conclusions take the form
"retention is sensitive to X and insensitive to Y" — a structural claim independent of
the specific values.

## Three methodological choices

**The metric counts circulation, not contact.** A common measure is whether a dollar ever
touches a local business. Under that measure, visitor → local bar → national supplier →
exit counts as a success. The primary metric here is the expected *number* of local
transactions per entering dollar. At baseline these differ substantially — about 48% of
dollars touch a local business, but the average dollar generates only about 1.18 local
transactions.

**The solution is analytic.** Expected visits to each transient state have a closed form
via the fundamental matrix `N = (I - Q)^-1`. Monte Carlo is included only to validate the
linear algebra; agreement to three decimals is expected, not a finding.

**Parameters are swept, not asserted.** The sensitivity analysis is the analysis.

## Finding

The entry split dominates everything downstream. The share of visitor spending captured
at the venue and official channels produces a swing roughly four times larger than any
other parameter, moving retention from about 0.75 to about 1.90 expected local
transactions across its plausible range. Every downstream recirculation parameter — local
wage share, household local spending, wholesaler sourcing — moves retention by only
10–20% individually.

This bears directly on the supplier programs several 2026 host cities implemented. Those
operate downstream, inserting local businesses into procurement chains after spending has
entered official channels. If this structure is roughly right, that is the lower-leverage
intervention. The higher-leverage question is whether spending enters official channels
at all — determined by venue design, event-day transport, ticketing bundles, and whether
the surrounding district has independent businesses visitors can reach.

A global sample of 20,000 draws across the full parameter space confirms the ordering;
parameter interactions do not change it.

## Contents

```
leakage.py                     Network specification, analytic solution, sensitivity tools
local_capture_analysis.ipynb   Analysis notebook with figures and discussion
requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
jupyter notebook local_capture_analysis.ipynb
```

Or use the module directly:

```python
import leakage as L

L.retention_metrics(L.BASELINE)     # analytic metrics at baseline
L.elasticity_table()                # what drives retention
L.global_sample(n=20_000)           # full parameter space
L.simulate(L.BASELINE)              # Monte Carlo validation
```

## Limitations

The network structure is assumed rather than estimated; the parameter sweep addresses
uncertainty in the values, not the topology. Nothing is calibrated against observed
spending. Dollars are treated as indivisible tokens, so the model cannot represent
proportional splitting within a single transaction. There is no time dimension, so the
step count is path length rather than velocity in the Fisher sense. Local wages are a
single node, so the model says nothing about who receives them — which is the question
that matters most for whether an event benefits a community.

## Next steps

1. Calibrate the entry split from transaction-level or card-panel data for a stadium
   district across event and non-event dates. That single parameter drives most of the
   variance.
2. Estimate downstream coefficients from BEA regional input-output accounts.
3. Add ownership structure — independent versus chain, resident versus non-resident.
4. Compare retention across 2026 host cities whose supplier programs differed, and
   instrument Los Angeles 2028 in advance rather than reconstructing it afterward.
