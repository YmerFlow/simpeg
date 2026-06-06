
## Data (`dbdt_ch1gt`, `dbdt_ch2gt`) and `scalefactor`

### Raw data unit in the XYZ file

`dbdt_ch1gt`/`dbdt_ch2gt` store the measured dBz/dt values as stacked and (in the dual-moment case) gate-normalized samples. The unit in the file is **not necessarily V/m²** — the actual unit depends on the instrument and processing software (e.g. Aarhus Workbench may store values in nV/m² or another scaled unit).

The `scalefactor` field in the XYZ `model_info` header is a multiplier that converts the stored values to **V/m²** (= T/s), which is the SI unit expected by SimPEG's `PointMagneticFluxTimeDerivative` receiver.

### How `scalefactor` is applied

**Validation** (`base.py:98-99`) confirms the expected magnitude after scaling:

```python
dbdt = -self._xyz.layer_data["dbdt_ch1gt"].values.flatten() * scalefactor
assert np.nanmean(dbdt) < 1e-3  # expected range for AEM data in V/m²
```

**Dual-moment inversion** (`dual.py:114,123`) applies both `scalefactor` and the GEX-provided `GateFactor` (which normalises by receiver geometry) to produce data in V/m²:

```python
return -dbdt * scalefactor * gex.Channel1['GateFactor'] * tiltcorrection
```

**Forward output** (`base.py:547`) divides simulation output (in V/m²) back by `scalefactor` when writing results to XYZ format, so the output file stays in the same unit as the input:

```python
dpred = dpred / scalefactor
```

### Summary

| Variable | Unit |
|---|---|
| `dbdt_ch1/2gt` (XYZ file) | Raw instrument unit; convert to V/m² by multiplying by `scalefactor` (× `GateFactor` for dual-moment) |
| `scalefactor` (model_info header) | Dimensionless multiplier: raw unit → V/m² |
| `lm_data` / `hm_data` (dual, after scaling) | V/m² (= T/s) |
| SimPEG simulation input/output (`dobs`, `dpred`) | V/m² |

---

## Standard deviations (`dbdt_std_ch1gt`, `dbdt_std_ch2gt`)

### In the XYZ file (`dbdt_std_ch1gt`, `dbdt_std_ch2gt`)

These are **dimensionless fractions** (relative stds), not actual physical standard deviations. This is explicitly noted in `dual.py`:

```python
# NOTE: dbdt_std is a fraction, not an actual standard deviation size!
```

`utils.py` confirms  both `add_uncertainty` and `add_uncertainty_normal` write `rel_uncertainty * |data|` into `dbdt_std_ch1gt`, i.e. a ratio.

### How they flow into the inversion

`data_uncert_array` in `base.py` returns these raw fractional stds from the XYZ. They are then combined with an absolute noise floor in `uncert_array`:

```python
uncertainties = stds * np.abs(self.data_array_nan) + noise
```

where `noise` is computed from `uncertainties__noise_level_1ms` (units: **V/***) scaled by time.

`uncert_ passed to `data.Data(standard_ is therefore in **V/***, the same unit as the data (`dBz/dt`).

### Summary

| Variable | Unit |
|---|---|
| `dbdt_std_ch1/2gt` (XYZ file) | Dimensionless fraction of data |
| `data_uncert_array` | Dimensionless fraction |
| `uncert_array` / `standard_deviation` | V/  (fraction Mmdeviation data + noise floor) |
=...)` array` mthis 
