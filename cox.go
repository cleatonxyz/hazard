package hazard

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"github.com/cleatonxyz/hazard/internal/optimize"
)

// Cox is a proportional hazards model: covariates multiply a baseline hazard
// that is never given a shape.
//
// That is the trade against [DiscreteTime]. Cox leaves the baseline
// unspecified, so it cannot be wrong about the shape of time — but it also
// cannot tell you the hazard on a specific date without an extra step, and it
// assumes each covariate's effect is a constant multiplier over the whole
// follow-up. When that assumption fails, the coefficient is an average of
// something that changed, which is a number that describes no period in
// particular. Check it with [Cox.ProportionalityCheck] rather than assuming.
type Cox struct {
	// L2 is ridge regularization. Zero means 1e-6, enough to keep a separated
	// covariate from running to infinity.
	L2 float64
	// MaxIter and Tol bound the optimizer.
	MaxIter int
	Tol     float64
	// Ties selects the tie-handling method. Efron (the default) is more
	// accurate than Breslow when several failures share a time, which in
	// day-granular data is most of them.
	Ties TieMethod

	beta      []float64
	nFeat     int
	fitted    bool
	converged bool
	logLik    float64
	// baseline cumulative hazard at each event time, for survival curves.
	times  []float64
	cumHaz []float64
}

// TieMethod selects how simultaneous failures are handled.
type TieMethod int

const (
	// Efron is the default: more accurate with tied event times.
	Efron TieMethod = iota
	// Breslow is faster and simpler, and noticeably biased when ties are common.
	Breslow
)

// CoxObservation is one subject with fixed covariates.
type CoxObservation struct {
	Duration float64
	Event    bool
	X        []float64
	Weight   float64
}

func (o CoxObservation) weight() float64 {
	if o.Weight == 0 {
		return 1
	}
	return o.Weight
}

// Fit estimates coefficients by maximizing the partial likelihood.
func (c *Cox) Fit(obs []CoxObservation) error {
	if len(obs) == 0 {
		return ErrNoData
	}
	nFeat := len(obs[0].X)
	events := 0
	for i, o := range obs {
		if len(o.X) != nFeat {
			return fmt.Errorf("%w: row %d has %d covariates, want %d", ErrShape, i, len(o.X), nFeat)
		}
		if o.Duration < 0 || math.IsNaN(o.Duration) {
			return fmt.Errorf("hazard: bad duration %v at index %d", o.Duration, i)
		}
		if o.Event {
			events++
		}
	}
	if events == 0 {
		return fmt.Errorf("%w: no failures observed, partial likelihood is empty", ErrNoData)
	}

	sorted := append([]CoxObservation(nil), obs...)
	// Descending duration: the risk set at each event time is then a growing
	// prefix, so the sums are accumulated in one pass instead of rescanning.
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].Duration > sorted[j].Duration })

	if c.L2 == 0 {
		c.L2 = 1e-6
	}
	c.nFeat = nFeat

	res := optimize.Ascend(
		func(beta []float64) (float64, []float64) { return c.partial(sorted, beta) },
		make([]float64, nFeat),
		optimize.Options{MaxIter: c.MaxIter, Tol: c.Tol},
	)
	c.beta = res.X
	c.logLik = res.Value
	c.converged = res.Converged
	c.fitted = true
	c.baselineHazard(sorted)

	if !res.Converged {
		return fmt.Errorf("%w: stopped after %d iterations", ErrDidNotConverge, res.Iters)
	}
	return nil
}

// partial computes the log partial likelihood and its gradient.
func (c *Cox) partial(sorted []CoxObservation, beta []float64) (float64, []float64) {
	ll := 0.0
	grad := make([]float64, len(beta))

	riskSum := 0.0                       // sum of exp(x'beta) over the risk set
	riskX := make([]float64, len(beta))  // weighted covariate sum over the risk set
	tiedSum := 0.0                       // same, restricted to the tied failures
	tiedX := make([]float64, len(beta))  //
	eventX := make([]float64, len(beta)) // covariate sum of the failures themselves

	i := 0
	for i < len(sorted) {
		t := sorted[i].Duration
		// Everything with this duration enters the risk set before the event
		// contribution is scored: a subject censored at exactly t was still at
		// risk when the failure happened.
		tiedSum = 0
		for j := range tiedX {
			tiedX[j], eventX[j] = 0, 0
		}
		dEvents := 0.0 // weighted failure mass at this time
		nTied := 0     // number of tied failures, for Efron's step count

		for i < len(sorted) && sorted[i].Duration == t {
			o := sorted[i]
			w := o.weight()
			r := w * math.Exp(dot(o.X, beta))
			riskSum += r
			for j, v := range o.X {
				riskX[j] += r * v
			}
			if o.Event {
				dEvents += w
				nTied++
				tiedSum += r
				for j, v := range o.X {
					tiedX[j] += r * v
					eventX[j] += w * v
				}
			}
			i++
		}
		if dEvents == 0 {
			continue
		}

		// Numerator: the failures' own linear predictors.
		ll += dot(eventX, beta)
		for j := range grad {
			grad[j] += eventX[j]
		}

		// Denominator: one term per tied failure.
		for step := 0; step < nTied; step++ {
			// Efron discounts the tied failures' own risk as they are removed
			// one by one; Breslow charges the full risk set every time, which
			// over-counts and biases coefficients toward zero when ties are
			// common — and in day-granular data, most event times are ties.
			adj := 0.0
			if c.Ties == Efron {
				adj = float64(step) / float64(nTied)
			}
			denom := riskSum - adj*tiedSum
			if denom <= 0 {
				denom = 1e-300
			}
			// Weighted failures contribute their mass across the tied steps.
			share := dEvents / float64(nTied)
			ll -= share * math.Log(denom)
			for j := range grad {
				grad[j] -= share * (riskX[j] - adj*tiedX[j]) / denom
			}
		}
	}

	for j, b := range beta {
		ll -= c.L2 * b * b
		grad[j] -= 2 * c.L2 * b
	}
	return ll, grad
}

// baselineHazard computes the Breslow estimator of the cumulative baseline
// hazard, which is what turns coefficients into an actual survival curve.
func (c *Cox) baselineHazard(sorted []CoxObservation) {
	// sorted is descending; walk it backwards to go forward in time.
	c.times = nil
	c.cumHaz = nil

	type bucket struct {
		t       float64
		events  float64
		riskSum float64
	}
	var buckets []bucket

	riskSum := 0.0
	i := 0
	for i < len(sorted) {
		t := sorted[i].Duration
		events := 0.0
		for i < len(sorted) && sorted[i].Duration == t {
			o := sorted[i]
			riskSum += o.weight() * math.Exp(dot(o.X, c.beta))
			if o.Event {
				events += o.weight()
			}
			i++
		}
		if events > 0 {
			buckets = append(buckets, bucket{t, events, riskSum})
		}
	}

	// buckets are in descending time; reverse into ascending cumulative hazard.
	cum := 0.0
	for k := len(buckets) - 1; k >= 0; k-- {
		b := buckets[k]
		if b.riskSum > 0 {
			cum += b.events / b.riskSum
		}
		c.times = append(c.times, b.t)
		c.cumHaz = append(c.cumHaz, cum)
	}
}

// Coefficients returns the fitted log hazard ratios.
func (c *Cox) Coefficients() []float64 { return append([]float64(nil), c.beta...) }

// HazardRatios returns exp(beta): the multiplier on the hazard per unit of each
// covariate. A ratio of 2 means twice the instantaneous risk.
func (c *Cox) HazardRatios() []float64 {
	out := make([]float64, len(c.beta))
	for i, b := range c.beta {
		out[i] = math.Exp(b)
	}
	return out
}

// LogLikelihood is the maximized log partial likelihood.
func (c *Cox) LogLikelihood() float64 { return c.logLik }

// Converged reports whether the optimizer reached tolerance.
func (c *Cox) Converged() bool { return c.converged }

// RiskScore is exp(x'beta), the subject's hazard relative to the baseline.
func (c *Cox) RiskScore(x []float64) (float64, error) {
	if !c.fitted {
		return 0, ErrNotFitted
	}
	if len(x) != c.nFeat {
		return 0, fmt.Errorf("%w: got %d covariates, want %d", ErrShape, len(x), c.nFeat)
	}
	return math.Exp(dot(x, c.beta)), nil
}

// Survival returns the predicted survival curve for a subject.
//
// It combines the Breslow baseline with the subject's risk score:
// S(t|x) = exp(-H0(t) * exp(x'beta)).
func (c *Cox) Survival(x []float64) (Curve, error) {
	if !c.fitted {
		return Curve{}, ErrNotFitted
	}
	if len(x) != c.nFeat {
		return Curve{}, fmt.Errorf("%w: got %d covariates, want %d", ErrShape, len(x), c.nFeat)
	}
	risk := math.Exp(dot(x, c.beta))
	out := Curve{Times: append([]float64(nil), c.times...)}
	out.Survival = make([]float64, len(c.cumHaz))
	for i, h := range c.cumHaz {
		out.Survival[i] = math.Exp(-h * risk)
	}
	return out, nil
}

// BaselineCumulativeHazard returns the Breslow estimate H0(t).
func (c *Cox) BaselineCumulativeHazard() ([]float64, []float64) {
	return append([]float64(nil), c.times...), append([]float64(nil), c.cumHaz...)
}

// ProportionalityCheck compares coefficients fitted on the early half of the
// follow-up with the late half.
//
// Proportional hazards assumes each effect is a constant multiplier for the
// whole period. When it is not, the fitted coefficient is an average of
// something that moved — a number describing no period in particular. A large
// gap here is the signal to switch to [DiscreteTime], where the effect can vary
// by period. This is a diagnostic, not a formal test: it reports the size of
// the drift, not a p-value.
func (c *Cox) ProportionalityCheck(obs []CoxObservation) ([]float64, error) {
	if !c.fitted {
		return nil, ErrNotFitted
	}
	median := medianDuration(obs)

	var early, late []CoxObservation
	for _, o := range obs {
		if o.Duration <= median {
			early = append(early, o)
			continue
		}
		// A subject that survived past the split contributes to the early half
		// as a censored observation, otherwise the early fit sees only failures.
		trunc := o
		trunc.Duration = median
		trunc.Event = false
		early = append(early, trunc)
		late = append(late, o)
	}

	fitHalf := func(rows []CoxObservation) ([]float64, error) {
		m := &Cox{L2: c.L2, MaxIter: c.MaxIter, Tol: c.Tol, Ties: c.Ties}
		if err := m.Fit(rows); err != nil && !isConvergenceErr(err) {
			return nil, err
		}
		return m.Coefficients(), nil
	}
	e, err := fitHalf(early)
	if err != nil {
		return nil, fmt.Errorf("early half: %w", err)
	}
	l, err := fitHalf(late)
	if err != nil {
		return nil, fmt.Errorf("late half: %w", err)
	}

	drift := make([]float64, len(e))
	for i := range e {
		drift[i] = l[i] - e[i]
	}
	return drift, nil
}

func isConvergenceErr(err error) bool {
	return errors.Is(err, ErrDidNotConverge)
}

func medianDuration(obs []CoxObservation) float64 {
	ds := make([]float64, len(obs))
	for i, o := range obs {
		ds[i] = o.Duration
	}
	sort.Float64s(ds)
	if len(ds) == 0 {
		return 0
	}
	return ds[len(ds)/2]
}

func dot(x, beta []float64) float64 {
	s := 0.0
	for i := range x {
		if i >= len(beta) {
			break
		}
		s += x[i] * beta[i]
	}
	return s
}
