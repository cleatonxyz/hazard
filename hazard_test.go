package hazard

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

func TestKaplanMeierMatchesHandComputedExample(t *testing.T) {
	// Textbook product-limit computation, worked by hand. The risk set at each
	// time is everyone with Duration >= t, which shrinks through censoring too:
	// t=1: 1 event of 6 at risk -> S = 5/6           = 0.8333
	// t=2: censored, no event; risk set drops to 4
	// t=3: 1 event of 4 at risk -> S = 0.8333 * 3/4  = 0.6250
	// t=4: censored, no event; risk set drops to 2
	// t=5: 2 events of 2 at risk -> S = 0.6250 * 0   = 0
	//
	// The curve reaching exactly 0 is correct, not a bug: the longest-observed
	// subject failed, so nothing is known to survive past t=5.
	obs := []Observation{
		{Duration: 1, Event: true},
		{Duration: 2, Event: false},
		{Duration: 3, Event: true},
		{Duration: 4, Event: false},
		{Duration: 5, Event: true},
		{Duration: 5, Event: true},
	}
	var km KaplanMeier
	if err := km.Fit(obs); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	c := km.Curve()

	want := []struct{ t, s float64 }{{1, 5.0 / 6}, {3, 0.625}, {5, 0}}
	if len(c.Times) != len(want) {
		t.Fatalf("got %d event times %v, want %d", len(c.Times), c.Times, len(want))
	}
	for i, w := range want {
		if c.Times[i] != w.t {
			t.Fatalf("time %d: got %v, want %v", i, c.Times[i], w.t)
		}
		if math.Abs(c.Survival[i]-w.s) > 1e-9 {
			t.Fatalf("S(%v): got %v, want %v", w.t, c.Survival[i], w.s)
		}
	}
}

func TestCensoringChangesTheAnswer(t *testing.T) {
	// The whole reason this package exists. Same durations, different censoring
	// flags: treating censored subjects as failures must not give the same
	// curve as handling them properly.
	durations := []float64{1, 2, 3, 4, 5, 6, 7, 8}

	allEvents := make([]Observation, len(durations))
	mixed := make([]Observation, len(durations))
	for i, d := range durations {
		allEvents[i] = Observation{Duration: d, Event: true}
		mixed[i] = Observation{Duration: d, Event: i%2 == 0}
	}

	var a, b KaplanMeier
	if err := a.Fit(allEvents); err != nil {
		t.Fatal(err)
	}
	if err := b.Fit(mixed); err != nil {
		t.Fatal(err)
	}

	// With half the subjects censored, survival must stay strictly higher than
	// when every exit is counted as a failure.
	if b.Curve().At(5) <= a.Curve().At(5) {
		t.Fatalf("censored fit S(5)=%v must exceed all-events S(5)=%v",
			b.Curve().At(5), a.Curve().At(5))
	}
}

func TestKaplanMeierRecoversKnownSurvival(t *testing.T) {
	// Geometric survival with per-period hazard 0.1: S(t) = 0.9^t.
	rng := rand.New(rand.NewSource(17))
	const h = 0.1
	obs := make([]Observation, 20000)
	for i := range obs {
		t := 1
		for rng.Float64() > h {
			t++
			if t > 200 {
				break
			}
		}
		obs[i] = Observation{Duration: float64(t), Event: true}
	}
	var km KaplanMeier
	if err := km.Fit(obs); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	for _, at := range []float64{5, 10, 20} {
		want := math.Pow(1-h, at)
		got := km.Curve().At(at)
		if math.Abs(got-want) > 0.02 {
			t.Fatalf("S(%v): got %.4f, want %.4f", at, got, want)
		}
	}
}

func TestHorizonAt(t *testing.T) {
	c := Curve{
		Times:    []float64{1, 2, 3, 4},
		Survival: []float64{0.95, 0.85, 0.6, 0.4},
	}
	tests := []struct {
		threshold float64
		want      float64
	}{
		{0.9, 1},  // still >= 0.9 at t=1, drops below at t=2
		{0.8, 2},  //
		{0.5, 3},  //
		{0.99, 0}, // already below at the first event time
		{0.3, 4},  // never falls below within the observed window
	}
	for _, tc := range tests {
		if got := c.HorizonAt(tc.threshold); got != tc.want {
			t.Fatalf("HorizonAt(%v) = %v, want %v", tc.threshold, got, tc.want)
		}
	}
	if !math.IsNaN(c.HorizonAt(0)) || !math.IsNaN(c.HorizonAt(1.5)) {
		t.Fatal("out-of-range thresholds must return NaN")
	}
}

func TestRestrictedMean(t *testing.T) {
	// S = 1 until t=1, then 0.5 until t=2, then 0.
	// Area to horizon 2 = 1*1 + 0.5*1 = 1.5
	c := Curve{Times: []float64{1, 2}, Survival: []float64{0.5, 0}}
	if got := c.RestrictedMean(2); math.Abs(got-1.5) > 1e-9 {
		t.Fatalf("got %v, want 1.5", got)
	}
	// Beyond the last event the curve stays at 0, so the area does not grow.
	if got := c.RestrictedMean(10); math.Abs(got-1.5) > 1e-9 {
		t.Fatalf("got %v, want 1.5", got)
	}
}

func TestDiscreteTimeRecoversCoefficientSign(t *testing.T) {
	// Two groups: x=1 fails roughly three times as often per period as x=0.
	// The fitted coefficient must be clearly positive.
	rng := rand.New(rand.NewSource(23))
	var rows []PersonPeriod
	for subj := 0; subj < 4000; subj++ {
		x := float64(subj % 2)
		h := 0.05
		if x == 1 {
			h = 0.15
		}
		for p := 1; p <= 20; p++ {
			failed := rng.Float64() < h
			rows = append(rows, PersonPeriod{Period: p, X: []float64{x}, Failed: failed})
			if failed {
				break
			}
		}
	}

	d := &DiscreteTime{}
	if err := d.Fit(rows); err != nil && !errors.Is(err, ErrDidNotConverge) {
		t.Fatalf("Fit: %v", err)
	}
	beta := d.Coefficients()
	if len(beta) != 1 {
		t.Fatalf("got %d coefficients, want 1", len(beta))
	}
	// True log odds ratio: log((0.15/0.85)/(0.05/0.95)) ~= 1.20
	if beta[0] < 0.9 || beta[0] > 1.5 {
		t.Fatalf("coefficient %v is far from the true log odds ratio ~1.20", beta[0])
	}

	h0, err := d.Hazard(1, []float64{0})
	if err != nil {
		t.Fatalf("Hazard: %v", err)
	}
	h1, _ := d.Hazard(1, []float64{1})
	if math.Abs(h0-0.05) > 0.02 {
		t.Fatalf("baseline hazard %v, want ~0.05", h0)
	}
	if math.Abs(h1-0.15) > 0.03 {
		t.Fatalf("treated hazard %v, want ~0.15", h1)
	}
}

func TestDiscreteTimeSurvivalCurveIsMonotone(t *testing.T) {
	rng := rand.New(rand.NewSource(29))
	var rows []PersonPeriod
	for subj := 0; subj < 1500; subj++ {
		x := rng.Float64()
		for p := 1; p <= 10; p++ {
			failed := rng.Float64() < 0.08+0.1*x
			rows = append(rows, PersonPeriod{Period: p, X: []float64{x}, Failed: failed})
			if failed {
				break
			}
		}
	}
	d := &DiscreteTime{}
	if err := d.Fit(rows); err != nil && !errors.Is(err, ErrDidNotConverge) {
		t.Fatalf("Fit: %v", err)
	}

	c, err := d.SurvivalConstant([]float64{0.5}, 10)
	if err != nil {
		t.Fatalf("SurvivalConstant: %v", err)
	}
	for i := 1; i < len(c.Survival); i++ {
		if c.Survival[i] > c.Survival[i-1] {
			t.Fatalf("survival rose from %v to %v at period %d", c.Survival[i-1], c.Survival[i], i+1)
		}
	}
	if c.Survival[len(c.Survival)-1] >= 1 || c.Survival[0] > 1 {
		t.Fatalf("survival out of range: %v", c.Survival)
	}
}

func TestTimeVaryingCovariateShowsShock(t *testing.T) {
	// Subjects are calm for 5 periods, then a shock flag turns on and the
	// per-period hazard jumps. A model with time-varying covariates should show
	// the jump; averaging the flag over follow-up would wash it out.
	rng := rand.New(rand.NewSource(31))
	var rows []PersonPeriod
	for subj := 0; subj < 3000; subj++ {
		for p := 1; p <= 10; p++ {
			shock := 0.0
			h := 0.03
			if p > 5 {
				shock, h = 1, 0.25
			}
			failed := rng.Float64() < h
			rows = append(rows, PersonPeriod{Period: p, X: []float64{shock}, Failed: failed})
			if failed {
				break
			}
		}
	}
	d := &DiscreteTime{}
	if err := d.Fit(rows); err != nil && !errors.Is(err, ErrDidNotConverge) {
		t.Fatalf("Fit: %v", err)
	}
	calm, err := d.Hazard(3, []float64{0})
	if err != nil {
		t.Fatal(err)
	}
	shocked, err := d.Hazard(7, []float64{1})
	if err != nil {
		t.Fatal(err)
	}
	if shocked < 3*calm {
		t.Fatalf("shock hazard %v should dwarf calm hazard %v", shocked, calm)
	}
}

func TestExpandPersonPeriod(t *testing.T) {
	obs := []Observation{
		{Duration: 3, Event: true},
		{Duration: 2, Event: false},
	}
	cov := [][]float64{{1}, {0}}
	rows, err := ExpandPersonPeriod(obs, cov)
	if err != nil {
		t.Fatalf("ExpandPersonPeriod: %v", err)
	}
	if len(rows) != 5 {
		t.Fatalf("got %d rows, want 5", len(rows))
	}
	// Only the failing subject's last row carries an event.
	failures := 0
	for _, r := range rows {
		if r.Failed {
			failures++
			if r.Period != 3 {
				t.Fatalf("failure recorded in period %d, want 3", r.Period)
			}
		}
	}
	if failures != 1 {
		t.Fatalf("got %d failures, want 1 — censored subjects must not produce events", failures)
	}
}

func TestConcordance(t *testing.T) {
	obs := []Observation{
		{Duration: 1, Event: true},
		{Duration: 5, Event: true},
		{Duration: 9, Event: true},
	}
	// Risk ordered exactly with observed failure order.
	if got, err := Concordance([]float64{0.9, 0.5, 0.1}, obs); err != nil || math.Abs(got-1) > 1e-9 {
		t.Fatalf("perfect ranking: got %v, err %v", got, err)
	}
	// Exactly reversed.
	if got, err := Concordance([]float64{0.1, 0.5, 0.9}, obs); err != nil || got != 0 {
		t.Fatalf("reversed ranking: got %v, err %v", got, err)
	}
	// A constant prediction demonstrates no skill and must score 0.5.
	if got, err := Concordance([]float64{0.5, 0.5, 0.5}, obs); err != nil || math.Abs(got-0.5) > 1e-9 {
		t.Fatalf("constant ranking: got %v, err %v", got, err)
	}
}

func TestConcordanceSkipsUndecidablePairs(t *testing.T) {
	// The censored subject left at t=1; whether it would have outlasted the
	// subject that failed at t=5 is unknowable, so the pair must not count.
	obs := []Observation{
		{Duration: 1, Event: false},
		{Duration: 5, Event: true},
	}
	if _, err := Concordance([]float64{0.9, 0.1}, obs); !errors.Is(err, ErrNoData) {
		t.Fatalf("got %v, want ErrNoData for zero comparable pairs", err)
	}
}

func TestBrierScore(t *testing.T) {
	obs := []Observation{
		{Duration: 2, Event: true},  // dead by horizon 5
		{Duration: 9, Event: true},  // alive at horizon 5
		{Duration: 1, Event: false}, // censored before horizon: excluded
	}
	// Perfect predictions on the two observable subjects.
	got, err := BrierScore([]float64{0, 1, 0.5}, obs, 5)
	if err != nil {
		t.Fatalf("BrierScore: %v", err)
	}
	if got != 0 {
		t.Fatalf("got %v, want 0 — the censored subject must be excluded", got)
	}
	// Worst possible predictions.
	got, err = BrierScore([]float64{1, 0, 0.5}, obs, 5)
	if err != nil {
		t.Fatalf("BrierScore: %v", err)
	}
	if math.Abs(got-1) > 1e-9 {
		t.Fatalf("got %v, want 1", got)
	}
}

func TestCalibrationDetectsOverconfidence(t *testing.T) {
	// Every subject is predicted to survive with p=0.9, but only half do.
	rng := rand.New(rand.NewSource(37))
	n := 2000
	obs := make([]Observation, n)
	pred := make([]float64, n)
	for i := range obs {
		pred[i] = 0.9
		if rng.Float64() < 0.5 {
			obs[i] = Observation{Duration: 2, Event: true} // dead before horizon 5
		} else {
			obs[i] = Observation{Duration: 20, Event: true}
		}
	}
	bins, err := Calibration(pred, obs, 5, 10)
	if err != nil {
		t.Fatalf("Calibration: %v", err)
	}
	if gap := MaxCalibrationGap(bins, 50); gap < 0.3 {
		t.Fatalf("expected a large calibration gap, got %v", gap)
	}
}

func TestErrorPaths(t *testing.T) {
	var km KaplanMeier
	if err := km.Fit(nil); !errors.Is(err, ErrNoData) {
		t.Fatalf("got %v, want ErrNoData", err)
	}
	d := &DiscreteTime{}
	if _, err := d.Hazard(1, []float64{0}); !errors.Is(err, ErrNotFitted) {
		t.Fatalf("got %v, want ErrNotFitted", err)
	}
	if err := d.Fit([]PersonPeriod{
		{Period: 1, X: []float64{1}, Failed: true},
		{Period: 1, X: []float64{1, 2}},
	}); !errors.Is(err, ErrShape) {
		t.Fatalf("got %v, want ErrShape", err)
	}
	if err := (&DiscreteTime{}).Fit([]PersonPeriod{
		{Period: 1, X: []float64{1}, Failed: false},
	}); !errors.Is(err, ErrNoData) {
		t.Fatalf("no failures should be refused, got %v", err)
	}
}
