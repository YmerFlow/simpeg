# Drop the deprecated `regularization__distance_weighted`

**State:** proposed 2026-11-03
**Tracking:** ymerflow-business#148
**Follows:** #38 (distance weighting always on, ymerflow-business#119)

## Where it stands

Since #38 the laterally constrained smoothness is always distance weighted. `regularization__distance_weighted` stays on `XYZSystem` only so that saved parameter sets that carry it still pass validation. aem-processes validates each job's system parameters against the schema generated from the class. That schema rejects unknown fields, so without the field a parameter set that carries the key would fail at job start.

How the field behaves now:

- `true` is ignored, with a `FutureWarning` that names this plan.
- `false` stops the run, saying distance weighting is always on.
- Unset, it does nothing.
- The schema marks it `deprecated: true`, so a form can hide it.

A job rerun in the environment it was made in is not affected: the environment pins the simpeg version, so the job reruns as it did. The field only matters where an old parameter set meets a newer environment:

- a process version cloned onto a newer environment;
- a template or forked workspace reused for a new job;
- a job definition kept in a script or an MCP job list that still sets the key.

## Before removal

1. **Inventory.** List every stored parameter set that still carries the key and could meet a newer environment. On the platform, the API lists processes and their versions, and a short script reading their system parameters gives the count. Job definitions kept outside the platform, such as scripts, notebooks and MCP job lists, have to be searched by whoever keeps them.
2. **Strip or flag.** Where the key is `true`, remove it. The result is identical, since `true` is what every environment from #38 on does anyway. Where it is `false`, the parameter set was tuned for the unweighted mode and its alphas don't carry over. It needs new lengths, so flag it to its owner rather than edit it mechanically.
3. **Hide it in the form.** Until removal, the form editor should hide fields marked `deprecated: true`, so no new parameter set picks the key up. Whether it does today needs checking on the frontend.

The stripping in step 2 could happen in three places:

- **A one-off script over the inventory, through the platform API** (recommended). It runs once and leaves nothing behind to maintain.
- **aem-processes dropping the key before validation.** This keeps old parameter sets working indefinitely, but moves the shim into aem-processes instead of removing it.
- **The platform rewriting parameters on clone.** This would make the platform know about a simpeg field, which it shouldn't.

## How long

Proposed: keep the field until both of these hold:

- the inventory script reports no stored parameter set with the key;
- an environment built from a simpeg that has the warning has been in use for at least a month, so anything still sending the key has shown the warning in its log.

Then remove the field in the next environment build. The dates are to be settled on ymerflow-business#148.

## What removal touches

- `simpeg/electromagnetics/utils/static_instrument/base.py`: the field and its docstring, and the check at the top of `smoothness_weights`, about 15 lines.
- `tests/em/static_instrument/test_inline_otherline_smoothness.py`: the two deprecation tests, and the `deprecated` assertion in the schema test.
- This plan: mark it done.

Nothing in aem-processes or the platform changes, because the schema is generated from the class. After removal, a parameter set that still carries the key fails schema validation at job start, as any unknown field does.
