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

## Cox proportional hazards

```go
c := &hazard.Cox{}            // Ties defaults to Efron
if err := c.Fit(observations); err != nil { return err }

c.Coefficients()              // log hazard ratios
c.HazardRatios()              // exp(beta): 2.0 means twice the risk
c.RiskScore(x)                // this subject relative to baseline
curve, _ := c.Survival(x)     // Breslow baseline + risk score
```

Cox never assumes a shape for the baseline hazard, so it cannot be wrong about
the shape of time. In exchange it assumes each covariate's effect is a constant
multiplier across the whole follow-up. When that fails, the coefficient is an
average of something that moved — a number describing no period in particular:

```go
drift, _ := c.ProportionalityCheck(observations)  // early-half vs late-half coefficients
```

A large drift is the signal to switch to `DiscreteTime`, where the effect is
free to vary by period.

**Ties matter more than they look.** With day-granular data most event times are
tied, and Breslow charges the full risk set for every tied failure, biasing
coefficients toward zero. Efron discounts them as they are removed and is the
default. The test suite checks that Efron is the more accurate of the two on
heavily tied data rather than taking it on faith.

## Confidence bands and comparisons

```go
band, _ := km.Greenwood(0.95)              // pointwise band on the log-log scale
na, _ := hazard.FitNelsonAalen(obs)        // cumulative hazard H(t)
res, _ := hazard.LogRank(groupA, groupB)   // do two groups differ?
fmt.Println(res)  // chi2=18.402 p=0.0000 (group A did worse than expected: ...)
```

Three things worth knowing about these:

**The band is pointwise, not simultaneous.** Each time point covers at the
stated level; the probability the *whole curve* stays inside is lower. Quoting
one as the other is the standard way to overstate certainty.

**It is built on the log-log scale.** A naive `S ± z·se` runs above 1 near the
start and below 0 in the tail, and a band containing impossible values reads as
a broken estimate.

**Log-rank is weakest exactly where curves cross.** It is most powerful under
proportional hazards; two curves that cross can return a large p-value while
being obviously different, because early and late differences cancel.

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
