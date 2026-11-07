---
name: neural-tensor-interventions
description: Design and test bounded tensor interventions on an existing neural model to investigate a measured behavior, with matched controls and independently recomputed metrics. Use when a user asks for a causal probe, activation intervention, or test of a model mechanism.
---

# Tensor interventions

Start from the observed behavior and its primary evidence. Identify the model, tensor, precise location relative to recurrence and decoding, available labels, and what the existing data can actually resolve. Consult applicable project instructions and record prior relevant tests; do not repeat completed experiments.

Before evaluating outcomes, fix a hypothesis, the intervention and its magnitude, matched original and identity controls, a sensible alternative intervention/control, inputs, split, metrics, failure conditions, provenance, and computational budget. Keep evaluation scenes separate from intervention selection when scientific inference is intended. Distinguish a one-case engineering check from an experiment that can support a scientific conclusion. If labels are used, restrict them to fitting and evaluation; never feed future frames or evaluation labels to the model.

Intervene on a cloned, explicitly indexed tensor; validate shape, dtype, device, finiteness, index uniqueness and range, unchanged unselected values, and no mutation of the original. Restore hooks and state after use. Preserve original checkpoints and data. Explicitly distinguish activation changes before a decoder from changes to recurrent state, parameters, input, or readout; a decoder-only intervention does not establish a recurrent-kernel mechanism. Inspect dependency-specific semantics and model evaluation mode before applying interventions. `scripts/intervention.py` supplies a small pure-tensor helper when appropriate; verify its assumptions against the model.

Run an identity intervention and compare outputs with the untouched reference, then evaluate the proposed intervention and a matched alternative on exactly the same inputs and initial conditions. Save raw outputs and exact metadata before interpretation. An independently implemented evaluator must read the saved outputs and recompute predefined metrics; document any shared code, labels, input generator, and numerical tolerance. Measure latency and errors together where relevant, including missing detections; do not select favorable outcomes after seeing them. Use a no-model or no-component comparator when useful.

Report the intervention's measured effect, uncertainty and limits, then distinguish causation within the specified intervention from the mechanism of the observed failure. A changed output alone is not an improvement or proof of the root cause. If a check fails, retain the failure and stop or create a separately versioned correction without weakening criteria. Do not silently continue into expensive runs or alter protected files.
