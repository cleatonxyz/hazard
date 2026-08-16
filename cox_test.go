package hazard

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

// exponentialCox draws subjects whose hazard is baseline*exp(beta*x), which is
// exactly the model Cox assumes — so a correct fit must recover beta.
func exponentialCox(rng *rand.Rand, n int, beta, baseline, censorAt float64) []CoxObservation {
	obs := make([]CoxObservation, n)
	for i := range obs {
		x := rng.NormFloat64()
		rate := baseline * math.Exp(beta*x)
		t := rng.ExpFloat64() / rate
		event := true
		if t > censorAt {
			t, event = censorAt, false
		}
		obs[i] = CoxObservation{Duration: t, Event: event, X: []float64{x}}
	}
	return obs
}

func TestCoxRecoversKnownCoefficient(t *testing.T) {
	rng := rand.New(rand.NewSource(401))
	const trueBeta = 0.8
	obs := exponentialCox(rng, 4000, trueBeta, 0.1, 40)

	c := &Cox{}
	if err := c.Fit(obs); err != nil && !errors.Is(err, ErrDidNotConverge) {
		t.Fatalf("Fit: %v", err)
	}
	got := c.Coefficients()[0]
	if math.Abs(got-trueBeta) > 0.08 {
		t.Fatalf("coefficient %.4f, want ~%.2f", got, trueBeta)
	}
	if hr := c.HazardRatios()[0]; math.Abs(hr-math.Exp(trueBeta)) > 0.2 {
		t.Fatalf("hazard ratio %.4f, want ~%.4f", hr, math.Exp(trueBeta))
	}
	if !c.Converged() {
		t.Log("note: optimizer hit its iteration cap")
	}
}

func TestCoxHandlesNegativeEffect(t *testing.T) {
	rng := rand.New(rand.NewSource(403))
	const trueBeta = -0.6
	obs := exponentialCox(rng, 4000, trueBeta, 0.1, 40)

	c := &Cox{}
	if err := c.Fit(obs); err != nil && !errors.Is(err, ErrDidNotConverge) {
		t.Fatal(err)
	}
	got := c.Coefficients()[0]
	if math.Abs(got-trueBeta) > 0.08 {
		t.Fatalf("coefficient %.4f, want ~%.2f", got, trueBeta)
	}
	if hr := c.HazardRatios()[0]; hr >= 1 {
		t.Fatalf("a protective covariate must give a ratio below 1, got %v", hr)
	}
}

func TestEfronBeatsBreslowWithHeavyTies(t *testing.T) {
	// Round durations to whole days so most event times are ties — the regime
	// where Breslow is known to shrink coefficients toward zero.
	rng := rand.New(rand.NewSource(405))
	const trueBeta = 1.0
	obs := exponentialCox(rng, 4000, trueBeta, 0.15, 20)
	for i := range obs {
		obs[i].Duration = math.Round(obs[i].Duration)
	}

	efron := &Cox{Ties: Efron}
	breslow := &Cox{Ties: Breslow}
	for _, m := range []*Cox{efron, breslow} {
		if err := m.Fit(obs); err != nil && !errors.Is(err, ErrDidNotConverge) {
			t.Fatal(err)
		}
	}
	eErr := math.Abs(efron.Coefficients()[0] - trueBeta)
	bErr := math.Abs(breslow.Coefficients()[0] - trueBeta)
	if eErr > bErr {
		t.Fatalf("Efron should be at least as accurate with ties: efron err %.4f vs breslow %.4f",
			eErr, bErr)
	}
	if breslow.Coefficients()[0] > efron.Coefficients()[0] {
		t.Fatalf("Breslow is expected to shrink toward zero: breslow %.4f vs efron %.4f",
			breslow.Coefficients()[0], efron.Coefficients()[0])
	}
}

func TestCoxSurvivalCurveIsMonotoneAndOrdered(t *testing.T) {
	rng := rand.New(rand.NewSource(407))
	obs := exponentialCox(rng, 2000, 0.9, 0.1, 30)

	c := &Cox{}
	if err := c.Fit(obs); err != nil && !errors.Is(err, ErrDidNotConverge) {
		t.Fatal(err)
	}

	lowRisk, err := c.Survival([]float64{-1})
	if err != nil {
		t.Fatalf("Survival: %v", err)
	}
	highRisk, err := c.Survival([]float64{1})
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i < len(lowRisk.Survival); i++ {
		if lowRisk.Survival[i] > lowRisk.Survival[i-1]+1e-12 {
			t.Fatalf("survival rose at index %d", i)
		}
	}
	// Higher covariate, higher hazard, lower survival — at every time.
	for i := range lowRisk.Survival {
		if highRisk.Survival[i] > lowRisk.Survival[i] {
			t.Fatalf("high-risk survival above low-risk at t=%v: %v vs %v",
				lowRisk.Times[i], highRisk.Survival[i], lowRisk.Survival[i])
		}
	}
	times, cum := c.BaselineCumulativeHazard()
	if len(times) != len(cum) || len(times) == 0 {
		t.Fatal("baseline hazard missing")
	}
	for i := 1; i < len(cum); i++ {
		if cum[i] < cum[i-1] {
			t.Fatal("cumulative hazard decreased")
		}
	}
}

func TestCoxRiskScore(t *testing.T) {
	rng := rand.New(rand.NewSource(409))
	c := &Cox{}
	if err := c.Fit(exponentialCox(rng, 1000, 0.5, 0.1, 30)); err != nil && !errors.Is(err, ErrDidNotConverge) {
		t.Fatal(err)
	}
	base, err := c.RiskScore([]float64{0})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(base-1) > 1e-9 {
		t.Fatalf("a zero covariate must score exactly 1, got %v", base)
	}
	high, _ := c.RiskScore([]float64{2})
	if high <= base {
		t.Fatalf("higher covariate should score higher: %v vs %v", high, base)
	}
}

func TestProportionalityCheckDetectsDrift(t *testing.T) {
	// An effect that only exists in the second half of follow-up violates
	// proportional hazards. The check should show a large drift; a genuinely
	// proportional dataset should not.
	rng := rand.New(rand.NewSource(411))

	proportional := exponentialCox(rng, 3000, 0.8, 0.08, 40)
	c1 := &Cox{}
	if err := c1.Fit(proportional); err != nil && !errors.Is(err, ErrDidNotConverge) {
		t.Fatal(err)
	}
	steady, err := c1.ProportionalityCheck(proportional)
	if err != nil {
		t.Fatalf("ProportionalityCheck: %v", err)
	}

	// Now a dataset where x matters only late.
	var drifting []CoxObservation
	for i := 0; i < 3000; i++ {
		x := rng.NormFloat64()
		early := rng.ExpFloat64() / 0.05 // no covariate effect
		late := rng.ExpFloat64() / (0.05 * math.Exp(1.5*x))
		t0 := early
		if early > 15 {
			t0 = 15 + late
		}
		event := true
		if t0 > 40 {
			t0, event = 40, false
		}
		drifting = append(drifting, CoxObservation{Duration: t0, Event: event, X: []float64{x}})
	}
	c2 := &Cox{}
	if err := c2.Fit(drifting); err != nil && !errors.Is(err, ErrDidNotConverge) {
		t.Fatal(err)
	}
	moved, err := c2.ProportionalityCheck(drifting)
	if err != nil {
		t.Fatal(err)
	}

	if math.Abs(moved[0]) <= math.Abs(steady[0]) {
		t.Fatalf("drifting data should show more drift: %.4f vs steady %.4f", moved[0], steady[0])
	}
}

func TestCoxValidation(t *testing.T) {
	c := &Cox{}
	if err := c.Fit(nil); !errors.Is(err, ErrNoData) {
		t.Fatalf("got %v, want ErrNoData", err)
	}
	if _, err := c.RiskScore([]float64{1}); !errors.Is(err, ErrNotFitted) {
		t.Fatalf("got %v, want ErrNotFitted", err)
	}
	if _, err := c.Survival([]float64{1}); !errors.Is(err, ErrNotFitted) {
		t.Fatalf("got %v, want ErrNotFitted", err)
	}
	if _, err := c.ProportionalityCheck(nil); !errors.Is(err, ErrNotFitted) {
		t.Fatalf("got %v, want ErrNotFitted", err)
	}
	err := c.Fit([]CoxObservation{
		{Duration: 1, Event: true, X: []float64{1}},
		{Duration: 2, Event: false, X: []float64{1, 2}},
	})
	if !errors.Is(err, ErrShape) {
		t.Fatalf("got %v, want ErrShape", err)
	}
	err = c.Fit([]CoxObservation{{Duration: 1, Event: false, X: []float64{1}}})
	if !errors.Is(err, ErrNoData) {
		t.Fatalf("all-censored data has an empty partial likelihood: got %v", err)
	}
	if err := c.Fit([]CoxObservation{{Duration: -1, Event: true, X: []float64{1}}}); err == nil {
		t.Fatal("negative duration must be rejected")
	}

	good := &Cox{}
	if err := good.Fit([]CoxObservation{
		{Duration: 1, Event: true, X: []float64{1}},
		{Duration: 2, Event: true, X: []float64{0}},
	}); err != nil && !errors.Is(err, ErrDidNotConverge) {
		t.Fatal(err)
	}
	if _, err := good.RiskScore([]float64{1, 2}); !errors.Is(err, ErrShape) {
		t.Fatalf("got %v, want ErrShape", err)
	}
}

func BenchmarkCoxFit(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	obs := exponentialCox(rng, 2000, 0.7, 0.1, 30)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c := &Cox{}
		_ = c.Fit(obs)
	}
}
