# Front gate and gate windows in the TEM forward

**State:** proposed, decisions open
**Ticket:** YmerFlow/Ymerflow#94 (the front gate, and the flat ~2% left after the receiver filters)
**Builds on:** #44 (the GEX receiver filter order read as AarhusInv reads it)
**Target system:** SkyTEM 312HP. **Validation:** the public SkyTEM 304 benchmark line in ymerflow-demo-data, our forward of the published Workbench model divided by Workbench's own forward of it.

## Summary

With #44 in place, the front gate explains almost nothing at the gates we keep. On the benchmark it moves one gate, the first kept high-moment gate at 72.7 us, by 1.6%, and no low-moment gate. On the SkyTEM 312HP it changes no kept gate on either moment, for any halfspace from 3 to 1000 ohm.m, as long as it sits after the coil filter, which is where AarhusInv puts it.

What #44 left on the benchmark has three other causes, each measured:

1. **The fork samples each gate at its center time; Workbench averages over the gate window.** This is most of the 1-1.5% low #44 left, and all of it from about 100 us on. Averaging over the window removes it.
2. **The transmitter is a point dipole.** The segmented loop the GEX describes gives 2.6% less at 8.5 us, 1.3% less at the first high-moment gate, and under 0.5% less late.
3. **The benchmark tool forwards at the measured altitude, Workbench at its inverted altitude.** That is a tool issue, not a fork issue. Fixing it cuts the sounding-to-sounding scatter of the ratio from about 5% to about 1.5%.

With all three, ours/Workbench is 0.987-1.003 at every low-moment gate from 8.5 us and 0.996-1.004 at every high-moment gate. On the SkyTEM 312HP, gate-window averaging matters more than anywhere else: up to 6% at kept low-moment gates.

The proposal: average over gate windows by adding rows to the fork's existing time matrix (no operator layer), offer the segmented loop behind an option until its cost is measured in an inversion, and leave the front gate out of the model, with a check that no kept gate sits close to it. A closed form for the front gate costs little in the same place and is decision 2.

## Measurements

### Method

Our forward is the fork at #44's head. It runs the published 40-layer model through the GEX the benchmark ships, with zero tilt, and divides by Workbench's forward of the same model per gate. Ratios are medians of the per-sounding ratio over 67 soundings (every 8th of the line's 531). "±" is half the 10-90 percentile spread across soundings.

To average over windows and to place a front gate between the filters, the response was computed on a dense time grid (about 1400 times per channel, 0.02 us steps around the front gate and 17 points across every gate window). The receiver chain was then applied in the time domain. The coil filter is a Gaussian-RLC and the board filter is first order on every SkyTEM GEX we have. For a first-order board filter with time constant tau, the chain coil filter, front gate at t_fg, board filter is exact in closed form:

    y(t) = 0                                   for t < t_fg
    y(t) = y0(t) - y0(t_fg) exp(-(t - t_fg)/tau)    for t >= t_fg

Here y0 is today's response with both filters applied everywhere in the frequency domain. tau is 1/(2 pi f), 0.53 us for the 304's 300 kHz board filter and 0.16 us for the 312HP's 1 MHz. So the front gate's whole effect dies out within about 5 tau of t_fg: 2.7 us on the 304 and 0.8 us on the 312HP. Only gates that open within that distance of the front gate can feel it.

### Benchmark

Per-sounding median of ours/Workbench, each column adding one change to the one before:

| Gate | #44 as it is (center, measured altitude) | + window average | + inverted altitude | + segmented loop | + front gate at FrontGateTime |
|---|---|---|---|---|---|
| LM 2.5 us | 1.098 ± 0.062 | 1.031 | 1.045 ± 0.015 | 1.027 | 1.027 |
| LM 4.5 us | 0.853 ± 0.048 | 0.901 | 0.911 ± 0.022 | 0.901 | 0.901 |
| LM 8.5 us | 1.019 ± 0.067 | 1.014 | 1.028 ± 0.020 | 1.003 | 1.003 |
| LM 10.5 us | 0.994 | 0.995 | 1.011 | 0.987 | 0.987 |
| LM 13 us | 0.991 | 0.995 | 1.010 | 0.988 | 0.988 |
| LM 21 us | 0.992 | 0.997 | 1.009 | 0.992 | 0.992 |
| LM 42 us | 0.991 | 0.998 | 1.009 | 0.996 | 0.996 |
| LM 109 us | 0.989 | 1.000 | 1.009 | 1.001 | 1.001 |
| LM 275 us | 0.985 | 1.001 | 1.007 | 1.002 | 1.002 |
| LM 696 us | 0.982 ± 0.018 | 1.004 | 1.002 ± 0.004 | 0.999 | 0.999 |
| HM 72.7 us | 1.005 ± 0.046 | 1.006 | 1.016 ± 0.012 | 1.004 | **0.988** |
| HM 76 us | 0.999 | 0.999 | 1.007 | 0.996 | 0.996 |
| HM 86 us | 0.999 | 1.000 | 1.010 | 0.999 | 0.999 |
| HM 102 us | 0.997 | 0.999 | 1.009 | 0.999 | 0.999 |
| HM 335 us | 0.988 | 0.999 | 1.004 | 0.999 | 0.999 |
| HM 3.6 ms | 0.977 ± 0.008 | 0.992 | 1.002 ± 0.003 | 1.000 | 1.000 |

The first column reproduces #44's table, which used all 531 soundings. The gates at 2.5 and 4.5 us are negative in Workbench's forward and in the data. The fork's secondary field reproduces the sign and roughly the size: -8.0e4 against -7.6e4 pV/(A m^4) at 2.5 us, window averaged. They lie inside the 304's 3.4 us turn-off ramp, the GEX removes them, and their remaining +3% and -10% have nothing to do with the front gate, which opens before the ramp starts (next section).

The last two columns were computed the way the proposal would compute them in the fork: five Gauss-Legendre nodes per window and the closed form above as fixed weights on the response at the nodes and at t_fg. They agree with the dense-grid columns to three decimals.

### The front-gate time

The scoping doc's main unknown was how the GEX's `FrontGateTime` (per channel) and `FrontGateDelay` (general) combine into AarhusInv's single front-gate time. On the benchmark GEX, `FrontGateTime` is -0.5 us on LM and 70 us on HM, `FrontGateDelay` is 2.5 us, and `GateTimeShift` is -1.75 us on LM and -2.0 us on HM. Each reading was applied to the dense-grid forward (point dipole, inverted altitude):

| Reading | LM, HM front gate | LM 2.5 us gate | HM 72.7 us gate | Verdict |
|---|---|---|---|---|
| no front gate | | 1.045 | 1.016 | 0.8% above the next HM gates |
| `FrontGateTime` | -0.5, 70.0 us | 1.045 | 1.001 | best named reading |
| `FrontGateTime + GateTimeShift` | -2.25, 68.0 us | 1.045 | 1.016 | same as no front gate |
| `FrontGateTime + FrontGateDelay + GateTimeShift` | 0.25, 70.5 us | 1.045 | 0.978 | 3% too low, ruled out |
| `FrontGateTime + FrontGateDelay` | 2.0, 72.5 us | 0.588 | 0.39 | ruled out |

The low moment can only rule out readings that put the gate later than about 1 us. Only the first kept high-moment gate (window 71.4-74.0 us) separates the rest. Matched to the gates after it, the best fit is 69.7 ± 0.2 us, 0.3 us before `FrontGateTime`. So Workbench does not add `FrontGateDelay` to the front-gate time, and `FrontGateTime` on its own fits to within 0.3 us. At that one gate the residual is ±0.8% either way, inside the data's 3% uniform STD. What `FrontGateDelay` does mean is still open.

On the SkyTEM 312HP GEX, `FrontGateDelay` and both `GateTimeShift`s are 0, so every reading gives the same front gate: -0.72 us on LM and 40 us on HM.

### Primary-field damping

The fork computes the secondary field only. AarhusInv computes the primary field of the segmented loop as well and multiplies it by `PrimaryFieldDampingFactor` (1e-6 LM, 0.01 HM on both GEX). The benchmark confirms that reading. At full strength, the loop's free-space primary would turn the 2.5 us gate from -8.0e4 into +1.6e5 pV/(A m^4), against Workbench's -7.6e4. With the damping factor it is below 1e-4 of the signal at every kept gate on both systems, because the receiver sits near the loop's null (its primary there is 2% of a point dipole's, and of opposite sign). For SkyTEM there is nothing to model.

### Gate sampling

The fork evaluates each gate at its center time. AarhusInv integrates over the gate when the open and close times are given (manual, version 8, p. 56), and the benchmark shows that Workbench does: the late gates go from 0.982-0.987 to 0.999-1.004 when we average over the window.

Gauss-Legendre quadrature over each window converges fast. Against a 33-point Simpson reference on the 312HP's kept gates, the worst error is 5e-2 with one node (the center), 6e-4 with two, 5e-5 with three and 1e-6 with five.

### SkyTEM 312HP

From the stock SkyTEM 312HP GEX in Ymerflow `gex/SkyTEM/`:

| | LM (Channel1) | HM (Channel2) |
|---|---|---|
| Coil filter (`RxCoilLPFilter1`) | 0.99, 48.7 kHz: Gaussian-RLC, about 4x lower than the 304's | same |
| Board filter (`TiBLowPassFilter`) | order 1, 1 MHz | order 1, 1 MHz |
| `FrontGateTime` / `FrontGateDelay` | -0.72 us / 0 | 40 us / 0 |
| `PrimaryFieldDampingFactor` | 1e-6 | 0.01 |
| `GateTimeShift`, `MeaTimeDelay` | 0, 0 | 0, 0 |
| Turn-off ramp | 22 us | 4.8 us |
| `NoGates`, `RemoveInitialGates` | 21, 6 | 40, 9 |
| First kept gate | 25.5 us (22.6-28.4) | 50.05 us (44.85-55.25) |
| `RepFreq` | 15 Hz | 15 Hz |
| `SystemResponseConvolution` | 0 | 0 |

The LM front gate opens before the turn-off starts, so it never blocks anything after time zero. The one HM gate that straddles 40 us (40.5 us, 36.2-44.8) is removed by `RemoveInitialGates`, and the HM gates before it lie entirely inside the closed front gate. `LoopType=73` is a segmented loop computed with the inverse Laplace transform rather than the Fourier transform (manual p. 50). It is required for a system response but does not imply one; this GEX is parametric. The scoping doc read it as a system-response marker.

Local forwards, halfspaces of 3, 10, 30, 100, 300 and 1000 ohm.m, 30 m altitude, point dipole unless stated. Each line is the range over halfspaces and kept gates of one change against the AarhusInv chain (coil filter, front gate, board filter, window average):

| Change | LM kept gates (25.5-114 us) | HM kept gates (50-254 us) |
|---|---|---|
| No front gate (today), window averaged | 1.0000 | 1.0000 |
| Front gate before the coil filter instead of after it | 1.0000 | 0.69-0.75 at 50 us, 0.97-0.98 at 62 us |
| Center sample instead of window average | 0.946-1.060 | 0.991-0.999 |
| Coil filter read as first order (before #44) | 0.55-1.40 | 0.89-0.99 |
| Segmented loop / point dipole (10-1000 ohm.m) | 0.966-0.998 | 0.988-0.999 |
| Primary field times the damping factor | below 1e-5 | below 1e-4 |

For the 312HP, then, the front gate needs no modeling. Wherever it is modeled, it has to come after the coil filter. Applied at the receiver input, before the filters, it would cut the first kept HM gate by 25-30%, because the 48.7 kHz coil filter takes 15-20 us to settle. What does matter is #44's coil-filter reading, averaging over the window, and, a few percent at the earliest kept LM gates, the loop.

## Design

### Gate windows as rows of the time matrix

The scoping doc proposed an operator layer: the response sampled densely through turn-off, and per channel a matrix to gate values. The fork already has that operator. `Simulation1DLayered._compute_freq_to_time_matricies` builds, per receiver, a matrix `A` from the frequency-domain response (receiver filters already applied there) to the receiver's times, with the waveform convolution inside. Forward and Jacobian both go through it. Averaging over a window is a fixed weighted sum of rows of `A`:

    A_gate[i] = 1/2 * sum_k w_k * A(node_ik)

with Gauss-Legendre nodes `node_ik` and weights `w_k` on the window [a_i, b_i]. Built that way, `A` keeps one row per gate, so the forward and Jacobian costs do not change. Only the one-time build sees more times: five nodes per gate is 5x today's, about 10 s once per run on the benchmark GEX.

- A receiver gets an optional `gate_windows` array, (n_times, 2) open and close in seconds, and `n_window_nodes`. Without them nothing changes.
- `MeasuredTEMXYZSystem.make_receiver` fills them from the GEX gate table (open and close, with `GateTimeShift` and `MeaTimeDelay` applied as for the centers), behind `simulation__gate_windows` (default on). Off reproduces today's results exactly.

### Front gate, if modeled (decision 2)

In the same construction, the closed form above is one more row: `A(t_fg)`. For gate i, the nodes go on [max(a_i, t_fg), b_i], and the row gets the term -tau (exp(-(a_i' - t_fg)/tau) - exp(-(b_i - t_fg)/tau)) / (b_i - a_i) times `A(t_fg)`. A gate that closes before t_fg is a zero row. That is exact for a first-order board filter, and every SkyTEM GEX we have uses one. Any other board filter would be refused with an error naming the GEX value, rather than approximated. About 20 lines beside the windows. The benchmark prototype above is exactly this.

### Front-gate check, if not modeled

At validation, warn when a kept gate opens less than 5 board-filter time constants after `FrontGateTime`, naming the channel and gate. On the 312HP the first kept gates open 30 time constants after the front gate, so it stays quiet. On the benchmark GEX the first kept HM gate opens 2.7 time constants after it and would be named: that is the gate the front gate moves by 1.6%.

### Segmented loop (decision 4)

`tdem.sources.LineCurrent` already works in `Simulation1DLayered`. For offset receivers, `make_system` would build it from `TxLoopPoint1..N` with current 1/`TxLoopArea` (unit moment), behind `simulation__tx_loop`. Three Gauss points per wire segment are enough for the secondary field: 1.5e-5 (LM) and 3e-4 (HM) against 20. The cost is the open question. On the 40-layer benchmark model a forward takes 6-8x as long as with the point dipole, because every wire point carries its own Hankel transform. Whether the stitched simulation and the Jacobian path accept a `LineCurrent` as they stand also has to be checked.

### Not proposed

- **Operator layer:** not needed, for the reason above. A measured system response (Ymerflow#95) is also a fixed linear map from the earth response to gate values, so it can go into the same matrix construction with its own kernel per channel; #95's own plan settles its units.
- **Primary field:** nothing to model for SkyTEM (above).
- **The gates inside the 304's turn-off ramp** (2.5 and 4.5 us): left removed, as the GEX says.

## Tests

- Windows: on the 312HP GEX and a halfspace, five nodes against a 33-point Simpson reference within 1e-5; `simulation__gate_windows=False` equal to today to machine precision; the Jacobian against finite differences with windows on.
- Front gate (if modeled): the closed-form rows against a time-domain simulation of coil filter, gate and board filter on a fine grid; a window that straddles the gate; a window that closes before it is zero; a board filter of order 2 is refused.
- Loop (if built): loop/dipole tends to 1 at late times; 3 against 20 wire points within 1e-3.
- Benchmark, locally, not committed: the table above.

## Decisions

| # | Question | Options | Suggested |
|---|---|---|---|
| 1 | Gate sampling | A: average over the gate window by Gauss-Legendre nodes, as rows of the time matrix, default on, option off. B: keep center sampling. | A, five nodes. It is the 1-1.5% at every gate on the benchmark and up to 6% at early 312HP gates, and costs nothing per forward. |
| 2 | The front gate | A: not modeled, plus the validation check. B: the closed form in the time matrix (first-order board filters only, others refused), default on. C: the scoping doc's operator layer. | A. It moves no kept gate on the 312HP, and on the benchmark it turns a +0.8% residual at one gate into -0.8%. B is the cheap upgrade if a system ever keeps gates within a few board time constants of its front gate. |
| 3 | Front-gate time (for the check, or for 2B) | `FrontGateTime` alone; `FrontGateTime + GateTimeShift`; with `FrontGateDelay`. | `FrontGateTime` alone. `FrontGateDelay` is ruled out on both moments; `GateTimeShift` can't be settled at better than 1% at one gate, and is 0 on the 312HP. |
| 4 | Transmitter for offset receivers | A: segmented loop behind `simulation__tx_loop`, default off until an inversion of the benchmark line is timed, then decide. B: loop default on now. C: keep the point dipole. | A. -2.6% at 8.5 us on the benchmark and up to -3.4% at the first kept 312HP LM gate, at 6-8x the forward cost. |
| 5 | The benchmark tool in ymerflow-demo-data | Forward at Workbench's inverted altitude, and average over windows once 1 lands. | Yes, as its own small PR there. |
| 6 | Order of work | | Windows first (one PR on top of #44, with the release note), then the loop option and its timing, then the check. |

Release note for 1: every forward and inversion result changes, by 1-1.5% at late gates on the benchmark and by up to 6% at the earliest kept 312HP gates.

## Effort

Focused working days with an AI coding assistant: gate windows with tests 0.5-1; the front-gate check 0.25 (or the closed form 0.5); the loop option, its tests and an inversion timing 1-1.5; the benchmark tool 0.25.
