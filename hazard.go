// Package hazard estimates how long something survives, and reports when it is
// likely to stop.
//
// It handles the fact that makes survival data different from ordinary
// regression: censoring. Most subjects under observation have not failed yet,
// and dropping them biases every estimate downward while treating their current
// age as a lifetime biases it the other way. Both are wrong; censoring is the
// correct treatment, and every estimator here does it.
//
// Two estimators, for two different questions:
//
//   - KaplanMeier: what does survival look like across a population, with no
//     covariates. Descriptive, non-parametric, few assumptions.
//   - DiscreteTime: how do covariates move the per-period risk of failure, and
//     what is the survival curve for one specific subject.
package hazard

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

var (
	// ErrNoData means the estimator was given an empty dataset.
	ErrNoData = errors.New("hazard: no observations")
	// ErrNotFitted means a model was queried before Fit.
	ErrNotFitted = errors.New("hazard: model not fitted")
	// ErrShape means covariate rows have inconsistent widths.
	ErrShape = errors.New("hazard: inconsistent covariate shape")
	// ErrDidNotConverge means the fit stopped without reaching tolerance.
	ErrDidNotConverge = errors.New("hazard: fit did not converge")
)

// Observation is one subject followed for Duration periods.
//
// Event is true if the subject failed at the end of that time, false if it was
// still alive when observation stopped (right-censored). Getting this flag
// wrong is the single most common way to produce a confident, wrong survival
// curve.
type Observation struct {
	Duration float64
	Event    bool
	// Weight lets one row stand for several identical subjects. Zero means 1.
	Weight float64
}

func (o Observation) weight() float64 {
	if o.Weight == 0 {
		return 1
	}
	return o.Weight
}

// Curve is a survival function, S(t) = P(still alive after t).
type Curve struct {
	Times    []float64 // strictly increasing
	Survival []float64 // non-increasing, starts at or below 1
	AtRisk   []float64 // subjects still under observation just before each time
	Events   []float64 // failures observed at each time
}

// At returns S(t) using the step function: the value carried forward from the
// last observed event time at or before t.
func (c Curve) At(t float64) float64 {
	if len(c.Times) == 0 || t < c.Times[0] {
		return 1
	}
	i := sort.SearchFloat64s(c.Times, t)
	if i == len(c.Times) || c.Times[i] > t {
		i--
	}
	return c.Survival[i]
}

// HorizonAt returns the largest t where S(t) is still at or above the given
// threshold: "how long can this hold before survival drops below p".
//
// This is the number a caller usually wants — not the probability at some fixed
// date, but the date at which a chosen probability runs out.
func (c Curve) HorizonAt(threshold float64) float64 {
	if threshold <= 0 || threshold > 1 {
		return math.NaN()
	}
	last := 0.0
	for i, s := range c.Survival {
		if s < threshold {
			return last
		}
		last = c.Times[i]
	}
	// Survival never fell below the threshold within the observed window; the
	// honest answer is bounded by how long we watched.
	return last
}

// MedianSurvival is HorizonAt(0.5), the conventional summary.
func (c Curve) MedianSurvival() float64 { return c.HorizonAt(0.5) }

// RestrictedMean is the area under S(t) up to horizon: expected lifetime,
// truncated at a point you actually observed.
//
// Prefer it to the plain mean. The mean of a survival distribution depends on
// the unobserved tail, so it is an extrapolation dressed as a summary.
func (c Curve) RestrictedMean(horizon float64) float64 {
	area, prev, prevS := 0.0, 0.0, 1.0
	for i, t := range c.Times {
		if t > horizon {
			break
		}
		area += prevS * (t - prev)
		prev, prevS = t, c.Survival[i]
	}
	if prev < horizon {
		area += prevS * (horizon - prev)
	}
	return area
}

// KaplanMeier is the non-parametric survival estimator.
type KaplanMeier struct{ curve Curve }

// Fit computes the product-limit estimate.
func (km *KaplanMeier) Fit(obs []Observation) error {
	if len(obs) == 0 {
		return ErrNoData
	}
	for i, o := range obs {
		if o.Duration < 0 || math.IsNaN(o.Duration) {
			return fmt.Errorf("hazard: bad duration %v at index %d", o.Duration, i)
		}
		if o.Weight < 0 {
			return fmt.Errorf("hazard: negative weight %v at index %d", o.Weight, i)
		}
	}

	sorted := append([]Observation(nil), obs...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Duration < sorted[j].Duration })

	total := 0.0
	for _, o := range sorted {
		total += o.weight()
	}

	var c Curve
	surv, atRisk, i := 1.0, total, 0
	for i < len(sorted) {
		t := sorted[i].Duration
		var events, censored float64
		for i < len(sorted) && sorted[i].Duration == t {
			if sorted[i].Event {
				events += sorted[i].weight()
			} else {
				censored += sorted[i].weight()
			}
			i++
		}
		if events > 0 {
			// Product-limit: each event time multiplies survival by the share
			// of the at-risk set that made it through.
			surv *= 1 - events/atRisk
			c.Times = append(c.Times, t)
			c.Survival = append(c.Survival, surv)
			c.AtRisk = append(c.AtRisk, atRisk)
			c.Events = append(c.Events, events)
		}
		// Censored subjects leave the risk set without contributing an event.
		atRisk -= events + censored
	}
	km.curve = c
	return nil
}

// Curve returns the fitted survival function.
func (km *KaplanMeier) Curve() Curve { return km.curve }
