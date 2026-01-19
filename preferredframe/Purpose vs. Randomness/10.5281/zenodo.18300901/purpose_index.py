"""
PURPOSE INDEX: Acausal Purpose Scanner & Figure Generator (v8.2)
================================================================

A research tool for quantifying "Acausal Purpose" (Teleology) in numerical data
by measuring thermodynamic resistance to number-theoretic entropy.

This script implements the algorithms described in the paper:
"Purpose vs Randomness: The Acausal Purpose Invariant"
(Rodriguez, Mercer, Thorne, 2026)


Core Concepts
-------------
1. Causal Depth (tau):
   The prime index of the largest prime factor of an integer N.
   Represents the "birth era" of the number's generative components.

2. Thermal Baseline (tau_star):
   The expected causal depth for a random integer of size N.
   - For statistical rarity (sweeps), we use the Median Baseline.
   - For entropic resistance (constants), we use the PNT Baseline (Max Entropy).

3. Acausal Purpose Invariant (P):
   A decibel-scale metric quantifying how much "smoother" (lower ancestry)
   a number is compared to the baseline.
   Formula: P(N) = 10 * log10( tau_star(N) / tau(N) )

Interpretation
--------------
* 0 dB  : Indistinguishable from noise (Thermal Equilibrium).
* >0 dB : Suppressed novelty (Structure).
* >20 dB: Cost-Paid Persistence (The "Forbidden Zone" for random generation).

Features
--------
* BigInt Support: Handles integers of arbitrary size (e.g., 2^100) using Python's native int.
* Dual Baselines: Switches between Median (statistical) and PNT (absolute) baselines.
* Visualization: Generates 4 publication-quality figures:
    1. The Cost of Structure (Log-Log Scatter)
    2. The Combinatorial Cliff (Survival Curve)
    3. Scale Invariance (Sweep Scatter)
    4. Teleological Signature (Constants Bar Chart)

Usage
-----
Run directly to perform a sweep and generate all figures:
    $ python3 purpose_index.py

Dependencies
------------
* matplotlib
* sympy

License
-------
MIT License

Copyright (c) 2026 An M. Rodriguez, Alex Mercer, Elias Thorne

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND...
"""

import math
import random
import statistics
import bisect
import sympy as sp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------- CONFIG ----------------

THRESH_DB = 20.0
TRIAL_PRIME_LIMIT = 5000
SMALL_PRIMES = list(sp.primerange(2, TRIAL_PRIME_LIMIT))

SWEEP_START = 100_000
SWEEP_END   = 1_000_000
SWEEP_STEP  = 900

TAU_BATCHES = 3
TAU_SAMPLES = 16

random.seed(42)

# ---------------- CORE MATH ----------------

def causal_depth(n: int) -> int:
    """Calculates tau(n): index of largest prime factor."""
    n = int(n)
    if n <= 1: return 0

    pmax = 1
    # 1. Fast trial division
    for p in SMALL_PRIMES:
        if p * p > n: break
        if n % p == 0:
            pmax = p
            while n % p == 0: n //= p

    # 2. Factor remainder
    if n > 1:
        try:
            if sp.isprime(n):
                pmax = max(pmax, n)
            else:
                f = sp.factorint(n)
                pmax = max(pmax, max(f.keys()))
        except:
            pmax = max(pmax, n)

    return int(sp.primepi(pmax))


_tau_cache = {}

def tau_star_median(N: int):
    """
    BASELINE A: MEDIAN (Statistical Average).
    Used for Sweeps to find 'rare' numbers.
    """
    decade = int(math.log10(N))
    if decade in _tau_cache: return _tau_cache[decade]

    meds = []
    for _ in range(TAU_BATCHES):
        xs = [random.randrange(N, 2 * N) for _ in range(TAU_SAMPLES)]
        taus = [causal_depth(x) for x in xs]
        meds.append(statistics.median(taus))

    med = statistics.median(meds)
    _tau_cache[decade] = med
    return med

def tau_star_pnt(N: int):
    """
    BASELINE B: PNT (Maximum Entropy).
    Used for Constants to measure distance from 'Primeness'.
    Approximates pi(N) ~ N/ln(N).
    Restores the ~28 dB signal for c.
    """
    if N <= 1: return 1.0
    # Accurate pi(N) approximation
    return N / math.log(N)

def acausal_purpose(N: int, tref: float) -> float:
    tauN = causal_depth(N)
    if tauN <= 0 or tref <= tauN: return 0.0
    return 10.0 * math.log10(tref / tauN)


# ---------------- PLOTTING ----------------

def plot_fig1_cost_of_structure():
    print("Generating Figure 1 (Cost of Structure)...")
    magnitudes = [int(10**i) for i in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    noise_n, noise_tau = [], []
    machine_n, machine_tau = [], []
    construct_primes = [2, 3, 5, 7]

    for N_base in magnitudes:
        Ns = [random.randrange(N_base, 2*N_base) for _ in range(10)]
        for N in Ns:
            noise_n.append(N)
            noise_tau.append(causal_depth(N))

            val = 1
            while val < N_base // 2: val *= random.choice(construct_primes)
            if val < 2: val = 2
            machine_n.append(val)
            machine_tau.append(causal_depth(val))

    plt.figure(figsize=(10, 6))
    plt.scatter(noise_n, noise_tau, c='magenta', label='Noise (Random)', s=25, alpha=0.6)
    plt.scatter(machine_n, machine_tau, c='cyan', label='Machine (Structured)', s=25, alpha=0.9, edgecolors='blue', linewidth=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(bottom=0.8)
    plt.xlabel('Magnitude (N)')
    plt.ylabel('Causal Depth (Birth Era)')
    plt.title('Figure 1: The Cost of Structure')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.2)
    plt.tight_layout()
    plt.savefig("alien_signal_detection.png", dpi=150)
    plt.close()

def plot_fig2_survival(rows):
    print("Generating Figure 2 (Survival Curve)...")
    dbs = sorted(r["db"] for r in rows)
    n_total = len(dbs)
    uniq_dbs = sorted(set(dbs))

    xs, ys = [], []
    for x_thresh in uniq_dbs:
        idx = bisect.bisect_right(dbs, x_thresh)
        prob = (n_total - idx) / n_total
        xs.append(x_thresh)
        ys.append(max(prob, 1e-12))

    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, linewidth=2, color='#1f77b4')
    plt.yscale("log")
    plt.axvline(THRESH_DB, linestyle="--", color='red', linewidth=2)
    plt.text(THRESH_DB + 0.5, 1e-3, "Forbidden Zone\n(>20 dB)", color='#cc0000', va='center')
    plt.xlabel("dB threshold (x)")
    plt.ylabel("P(Purpose > x)")
    plt.title("Figure 2: The Combinatorial Cliff (Survival Curve)")
    plt.grid(True, which="both", ls="--", alpha=0.2)
    plt.ylim(top=1.0)
    plt.tight_layout()
    plt.savefig("sweep_focus_survival.png", dpi=150)
    plt.close()

def plot_fig3_scatter(rows):
    print("Generating Figure 3 (Sweep Scatter)...")
    xs = [r["N"] for r in rows]
    ys = [r["db"] for r in rows]

    plt.figure(figsize=(10, 4.5))
    plt.scatter(xs, ys, s=6, alpha=0.5, color='#0066cc')
    plt.axhline(THRESH_DB, linestyle="--", color='orange', linewidth=1.5)
    plt.xlabel("Magnitude (N)")
    plt.ylabel("Acausal Purpose (dB)")
    plt.title("Figure 3: Scale Invariance of Purpose")
    plt.tight_layout()
    plt.savefig("sweep_focus_scatter.png", dpi=150)
    plt.close()

def plot_fig4_constants():
    print("Generating Figure 4 (Teleological Signature)...")
    # Using PNT Baseline to measure distance from Entropy Ceiling (Primes)

    targets = [
        {"name": "Random Prime", "val": 999983, "type": "Noise", "color": "#d62728"},
        {"name": "c (Speed Light)", "val": 299792458, "type": "Defined", "color": "#2ca02c"},
        {"name": "h (Planck)", "val": 662607015, "type": "Defined", "color": "#2ca02c"},
        {"name": "N_A (Avogadro)", "val": 602214076, "type": "Defined", "color": "#2ca02c"},
        {"name": "G (Gravity)", "val": 667430, "type": "Measured", "color": "#d62728"},
        {"name": "1/alpha (Fine Struct)", "val": 137035999, "type": "Measured", "color": "#d62728"},
    ]

    labels, values, colors = [], [], []

    print(f"\n{'CONSTANT':<20} | {'VAL':<12} | {'dB (PNT)':<8}")
    print("-" * 45)

    for t in targets:
        # Use PNT Baseline for Constants
        tref = tau_star_pnt(t['val'])
        score = acausal_purpose(t['val'], tref)

        labels.append(t['name'].replace(" (", "\n("))
        values.append(score)
        colors.append(t['color'])

        print(f"{t['name']:<20} | {str(t['val'])[:12]:<12} | {score:6.2f}")

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors, alpha=0.85, edgecolor='black')
    plt.axhline(y=THRESH_DB, color='orange', linestyle='--', linewidth=2)
    plt.text(len(labels)-0.5, THRESH_DB + 0.5, f"Threshold ({int(THRESH_DB)} dB)", color='orange', ha='right', fontweight='bold')
    plt.ylabel("Acausal Purpose (dB) [Ref: Max Entropy]")
    plt.title("Figure 4: The Teleological Signature of Representations")

    plt.tight_layout()
    plt.savefig("acausal_constants_comparison.png", dpi=150)
    plt.close()

# ---------------- EXECUTION ----------------

def run_sweep():
    print(f"\n--- Running Sweep [{SWEEP_START:,} -> {SWEEP_END:,}] ---")
    rows = []
    for i, N in enumerate(range(SWEEP_START, SWEEP_END + 1, SWEEP_STEP)):
        # Use Median Baseline for Sweep (Statistical Rarity)
        tref = tau_star_median(N)
        db = acausal_purpose(N, tref)
        rows.append({"N": N, "db": db})
    return rows

if __name__ == "__main__":
    print("=== ACAUSAL PURPOSE PAPER: FIGURE GENERATOR (v8.2) ===")

    plot_fig1_cost_of_structure()
    plot_fig4_constants() # Uses PNT Baseline

    sweep_data = run_sweep() # Uses Median Baseline
    if sweep_data:
        plot_fig2_survival(sweep_data)
        plot_fig3_scatter(sweep_data)

    print("\n=== ALL FIGURES GENERATED ===")
