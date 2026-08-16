package hazard

import (
	"fmt"
	"math"

	"github.com/cleatonxyz/hazard/internal/optimize"
)

// PersonPeriod is one subject in one time period: the row layout a discrete
// time hazard model consumes.
//
// A subject followed for 5 periods and failing in the 5th becomes 5 rows, with
// Failed true only on the last. Splitting a subject this way is what allows
// covariates to change over time — an incentive campaign that expires halfway
// through is a different X in later rows, not an averaged-out constant.
type PersonPeriod struct {
	// Period is the index within the subject's own follow-up, starting at 1.
	Period int
	// X holds covariates for this subject in this period.
	X []float64
	// Failed is true when the subject failed at the end of this period.
	Failed bool
	// Weight is optional; zero means 1.
	Weight float64
}

func (p PersonPeriod) weight() float64 {
	if p.Weight == 0 {
		return 1
	}
	return p.Weight
}

// DiscreteTime is a logistic hazard model: the probability of failing in each
// period, given survival up to it.
//
// Time itself enters through per-period baseline terms rather than a smooth
// function, so a shock at a known date (a campaign ending, a lockup expiring)
// shows as a spike instead of being smoothed away.
type DiscreteTime struct {
	// MaxPeriod is the largest period with its own baseline. Later periods
	// reuse the last one.
	MaxPeriod int
	// L2 is ridge regularization on covariate coefficients. Baselines are not
	// penalized. Defaults to 1e-6, enough to keep separation from exploding.
	L2 float64
	// MaxIter and Tol bound the optimizer.
	MaxIter int
	Tol     float64

	baseline []float64 // per-period intercept, index 0 == period 1
	beta     []float64 // covariate coefficients
	nFeat    int
	fitted   bool
}

// Coefficients returns the fitted covariate coefficients on the log-odds scale.
// A coefficient of 0.7 means a one-unit increase multiplies the odds of failing
// in a period by about 2.
func (d *DiscreteTime) Coefficients() []float64 {
	return append([]float64(nil), d.beta...)
}

// Baseline returns the fitted per-period intercepts.
func (d *DiscreteTime) Baseline() []float64 {
	return append([]float64(nil), d.baseline...)
}

// Fit estimates baselines and coefficients by penalized maximum likelihood.
//
// The optimizer is gradient ascent with backtracking, chosen over Newton's
// method because the Hessian is (P+K)x(P+K) and the baseline block makes it
// near-singular whenever a period has few events — a situation that is normal,
// not exceptional, in short panels.
func (d *DiscreteTime) Fit(rows []PersonPeriod) error {
	if len(rows) == 0 {
		return ErrNoData
	}
	nFeat := len(rows[0].X)
	maxPeriod := 0
	for i, r := range rows {
		if len(r.X) != nFeat {
			return fmt.Errorf("%w: row %d has %d covariates, want %d", ErrShape, i, len(r.X), nFeat)
		}
		if r.Period < 1 {
			return fmt.Errorf("hazard: period must be >= 1, got %d at row %d", r.Period, i)
		}
		if r.Period > maxPeriod {
			maxPeriod = r.Period
		}
	}
	if d.MaxPeriod > 0 && d.MaxPeriod < maxPeriod {
		maxPeriod = d.MaxPeriod
	}
	d.MaxPeriod = maxPeriod
	d.nFeat = nFeat
	if d.L2 == 0 {
		d.L2 = 1e-6
	}
	if d.MaxIter == 0 {
		d.MaxIter = 500
	}
	if d.Tol == 0 {
		d.Tol = 1e-8
	}

	d.baseline = make([]float64, maxPeriod)
	d.beta = make([]float64, nFeat)
	// Start baselines at the log-odds of the pooled failure rate so the
	// optimizer does not spend its first iterations finding the intercept.
	var events, total float64
	for _, r := range rows {
		w := r.weight()
		total += w
		if r.Failed {
			events += w
		}
	}
	if events == 0 {
		return fmt.Errorf("%w: no failures observed, nothing to fit", ErrNoData)
	}
	init := math.Log(events / math.Max(total-events, 1e-12))
	for i := range d.baseline {
		d.baseline[i] = init
	}

	// Parameters are packed as [baselines..., beta...] so the shared optimizer
	// sees one vector. The packing lives here rather than in optimize because
	// only this model knows which slots mean what.
	res := optimize.Ascend(
		func(x []float64) (float64, []float64) {
			d.unpack(x)
			gBase, gBeta := d.gradient(rows)
			return d.logLikelihood(rows), append(gBase, gBeta...)
		},
		append(append([]float64(nil), d.baseline...), d.beta...),
		optimize.Options{MaxIter: d.MaxIter, Tol: d.Tol},
	)
	d.unpack(res.X)
	d.fitted = true
	if !res.Converged {
		return fmt.Errorf("%w: stopped after %d iterations", ErrDidNotConverge, res.Iters)
	}
	return nil
}

// unpack splits the flat parameter vector back into baselines and coefficients.
func (d *DiscreteTime) unpack(x []float64) {
	copy(d.baseline, x[:len(d.baseline)])
	copy(d.beta, x[len(d.baseline):])
}

// Hazard returns the probability of failing in the given period, conditional on
// having survived to it.
func (d *DiscreteTime) Hazard(period int, x []float64) (float64, error) {
	if !d.fitted {
		return 0, ErrNotFitted
	}
	if len(x) != d.nFeat {
		return 0, fmt.Errorf("%w: got %d covariates, want %d", ErrShape, len(x), d.nFeat)
	}
	return sigmoid(d.linear(period, x)), nil
}

// Survival builds the survival curve for one subject over the given periods.
//
// covariates[i] holds the covariates for period i+1, letting them change over
// time. Passing a single row applies it to every period.
func (d *DiscreteTime) Survival(covariates [][]float64) (Curve, error) {
	if !d.fitted {
		return Curve{}, ErrNotFitted
	}
	if len(covariates) == 0 {
		return Curve{}, fmt.Errorf("%w: no covariates given", ErrShape)
	}
	var c Curve
	s := 1.0
	for i := range covariates {
		x := covariates[i]
		if len(x) != d.nFeat {
			return Curve{}, fmt.Errorf("%w: row %d has %d covariates, want %d", ErrShape, i, len(x), d.nFeat)
		}
		h := sigmoid(d.linear(i+1, x))
		s *= 1 - h
		c.Times = append(c.Times, float64(i+1))
		c.Survival = append(c.Survival, s)
	}
	return c, nil
}

// SurvivalConstant is Survival for a subject whose covariates never change.
func (d *DiscreteTime) SurvivalConstant(x []float64, periods int) (Curve, error) {
	rows := make([][]float64, periods)
	for i := range rows {
		rows[i] = x
	}
	return d.Survival(rows)
}

func (d *DiscreteTime) linear(period int, x []float64) float64 {
	idx := period - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(d.baseline) {
		idx = len(d.baseline) - 1 // periods past the fitted window reuse the last baseline
	}
	z := d.baseline[idx]
	for j, v := range x {
		z += d.beta[j] * v
	}
	return z
}

func (d *DiscreteTime) logLikelihood(rows []PersonPeriod) float64 {
	ll := 0.0
	for _, r := range rows {
		z := d.linear(r.Period, r.X)
		w := r.weight()
		if r.Failed {
			ll += w * logSigmoid(z)
		} else {
			ll += w * logSigmoid(-z)
		}
	}
	for _, b := range d.beta {
		ll -= d.L2 * b * b
	}
	return ll
}

func (d *DiscreteTime) gradient(rows []PersonPeriod) (gBase, gBeta []float64) {
	gBase = make([]float64, len(d.baseline))
	gBeta = make([]float64, len(d.beta))
	for _, r := range rows {
		z := d.linear(r.Period, r.X)
		p := sigmoid(z)
		y := 0.0
		if r.Failed {
			y = 1
		}
		resid := (y - p) * r.weight()

		idx := r.Period - 1
		if idx >= len(gBase) {
			idx = len(gBase) - 1
		}
		gBase[idx] += resid
		for j, v := range r.X {
			gBeta[j] += resid * v
		}
	}
	for j, b := range d.beta {
		gBeta[j] -= 2 * d.L2 * b
	}
	return gBase, gBeta
}

func sigmoid(z float64) float64 {
	if z >= 0 {
		return 1 / (1 + math.Exp(-z))
	}
	e := math.Exp(z)
	return e / (1 + e)
}

// logSigmoid computes log(1/(1+exp(-z))) without overflowing for large |z|,
// which the naive form does as soon as a covariate is poorly scaled.
func logSigmoid(z float64) float64 {
	if z >= 0 {
		return -math.Log1p(math.Exp(-z))
	}
	return z - math.Log1p(math.Exp(z))
}

// ExpandPersonPeriod turns subject-level records into person-period rows.
//
// Each subject contributes one row per period survived; Failed is set on the
// final row only when the subject actually failed, so censored subjects
// contribute exposure without a phantom event.
func ExpandPersonPeriod(obs []Observation, covariates [][]float64) ([]PersonPeriod, error) {
	if len(obs) != len(covariates) {
		return nil, fmt.Errorf("%w: %d observations vs %d covariate rows", ErrShape, len(obs), len(covariates))
	}
	var rows []PersonPeriod
	for i, o := range obs {
		periods := int(math.Round(o.Duration))
		if periods < 1 {
			continue // observed for less than one period: contributes nothing
		}
		for p := 1; p <= periods; p++ {
			rows = append(rows, PersonPeriod{
				Period: p,
				X:      covariates[i],
				Failed: o.Event && p == periods,
				Weight: o.Weight,
			})
		}
	}
	if len(rows) == 0 {
		return nil, ErrNoData
	}
	return rows, nil
}
