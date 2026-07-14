# Tiered-stress-biochemistry-model
## Feedback Sensitivity and Resource Depletion in Stress Biochemistry
The Tiered Stress Biochemistry Model (TSBM) is a reproducible ten-equation ordinary differential equation model developed to investigate how acute and sustained stress-related cortisol exposure may propagate through biochemical pathways operating on different timescales.

The model couples:

- locus coeruleus noradrenergic activity
- vitamin C utilization
- aldosterone-associated magnesium loss
- BDNF suppression
- Nrf2 suppression
- inflammatory activation
- tryptophan-to-kynurenine metabolism
- a slow stress-load state representing persistent endocrine burden

The model is intended as a hypothesis-generating computational framework. The simulated normal, acute stress, chronic stress, depression-like, and PTSD-like conditions are stylized model scenarios and should not be interpreted as diagnostic representations of individual patients.

## Scientific objective

TSBM was developed to examine why similar stress-related endocrine signals may lead to different downstream biochemical outcomes.

The model evaluates two broad simulated patterns:

1. **Resource-depletion trajectories**, characterized by progressive magnesium and vitamin C depletion, reduced BDNF and Nrf2, increased inflammation, and elevation of the kynurenine-to-tryptophan ratio.

2. **Feedback-sensitive trajectories**, in which lower cortisol exposure or stronger feedback limits downstream biochemical disruption.

The framework generates experimentally testable predictions regarding:

- the temporal order of biochemical changes
- early magnesium vulnerability
- later BDNF and kynurenine-pathway changes
- the influence of persistent slow stress-load
- the contribution of individual model pathways
- potential pre-threshold experimental intervention windows

These predicted windows are model-dependent hypotheses and are not established clinical treatment deadlines.

## Model structure

The state vector is:

```text
[NE, VitC, Ald, Mg, BDNF, Nrf2, INF, Trp, Kyn, Cstress]
