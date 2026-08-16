# hazard

Survival analysis in Go, with no dependencies.

Estimate how long something lasts, and get back the number people actually
want: *how many periods until survival drops below a threshold.*

> **Status: v0.** The API can still change.

## The thing that makes this different from regression

Most subjects have not failed yet when you look. Drop them and every estimate is
biased downward; count their current age as a finished lifetime and it is biased
the other way. Both are wrong. Handling that correctly — censoring — is what
every estimator here does, and it is why `mean(duration)` is not an answer.

## Install

```bash
go get github.com/cleatonxyz/hazard
```

## Kaplan–Meier: population survival, no covariates

```go
var km hazard.KaplanMeier
if err := km.Fit(observations); err != nil { // Duration + Event (false = censored)
    return err
}
c := km.Curve()

c.At(30)              // S(30): share still alive after 30 periods
c.HorizonAt(0.8)      // last period where survival is still >= 80%
c.MedianSurvival()    // HorizonAt(0.5)
c.RestrictedMean(60)  // expected lifetime, truncated at a horizon you observed
```

## Discrete-time hazard: covariates, and shocks on known dates

```go
rows, err := hazard.ExpandPersonPeriod(observations, covariates)
if err != nil {
    return err
}

d := &hazard.DiscreteTime{}
if err := d.Fit(rows); err != nil {
    return err
}

d.Coefficients()                        // log-odds effect per covariate
d.Hazard(7, x)                          // P(fail in period 7 | survived to 7)
curve, err := d.SurvivalConstant(x, 30) // full curve for one subject
```

Time enters as a per-period baseline rather than a smooth function, so a shock
on a known date — a campaign ending, a lockup expiring — shows up as a spike
instead of being smoothed into the trend. Covariates may vary by period, which
is the point of the person-period layout: pass a different `X` for the periods
after the shock rather than averaging it away.

Runnable version: `go run ./examples/horizon`.

## Evaluation

```go
hazard.Concordance(risk, obs)                    // Harrell's C: ranking skill
hazard.BrierScore(predSurvival, obs, horizon)    // squared error at a horizon
hazard.Calibration(predSurvival, obs, horizon, 10)
hazard.MaxCalibrationGap(bins, 50)
```

Discrimination and calibration are different properties, and a model can be
excellent at one while being useless at the other. A model that ranks subjects
perfectly (C near 1.0) can still put every probability far from the truth. If
the number gets published, check both.

## Deliberate limits

**Concordance skips undecidable pairs.** If a censored subject left before the
other failed, nobody knows who would have lasted longer. Counting that pair
either way manufactures skill that was never demonstrated, so it is excluded —
and if no pair is comparable, you get an error rather than a number.

**`BrierScore` uses the simple form.** Subjects censored before the horizon are
excluded. That is unbiased when censoring is independent of risk and biased when
it is not. If subjects leave your panel *because* they were about to fail, this
number flatters the model; IPCW weighting is the fix and is not implemented yet.

**The optimizer is gradient ascent with backtracking**, not Newton–Raphson. The
Hessian's baseline block is near-singular whenever a period has few events,
which in short panels is normal rather than exceptional. Slower, does not blow
up.

## Roadmap

- Cox proportional hazards with Efron ties
- competing risks (cumulative incidence)
- IPCW-weighted Brier score
- confidence bands (Greenwood)

## License

MIT
