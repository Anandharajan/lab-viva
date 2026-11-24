"""Example lab manual text and simulated students for IRT tests."""
from __future__ import annotations

import numpy as np

EXAMPLE_MANUAL = """Experiment 3: Half-wave and full-wave rectifier.
Aim: To construct a half-wave rectifier and a full-wave rectifier and measure the ripple factor.
Apparatus required: transformer, diode, resistor, capacitor, breadboard, oscilloscope and multimeter.
Theory: A half-wave rectifier conducts only during the positive half cycle of the input.
A full-wave rectifier conducts during both half cycles using two diodes and a centre-tapped transformer.
The ripple factor is the ratio of the RMS value of the AC component to the DC value of the output.
For a half-wave rectifier the ripple factor r = 1.21 without a filter capacitor.
For a full-wave rectifier the ripple factor r = 0.482 without a filter capacitor.
A filter capacitor across the load resistor reduces the ripple factor of the output.
The efficiency of the half-wave rectifier is at most 40.6 percent.
Procedure: Connect the circuit on the breadboard as shown in the circuit diagram.
Observe the input and output waveforms on the oscilloscope.
Measure the DC output voltage using the multimeter.
Repeat the measurement with the filter capacitor connected across the load resistor.
Compute the ripple factor from the measured AC and DC values.
"""


def simulate(n_students: int = 300, n_items: int = 25, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 1, n_students)
    a = rng.lognormal(0, 0.3, n_items)
    b = rng.uniform(-2, 2, n_items)
    P = 1 / (1 + np.exp(-a * (theta[:, None] - b)))
    R = (rng.random(P.shape) < P).astype(float)
    return {"theta": theta, "a": a, "b": b, "R": R}
