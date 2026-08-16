package hazard

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

func TestGreenwoodStaysInsideZeroOne(t *testing.T) {
	// The reason for the log-log scale: a naive S ± z·se produces bounds above
	// 1 near the start and below 0 in the tail, and a band containing
	// impossible values reads as a broken estimate.
	rng := rand.New(rand.NewSource(501))
	obs := make([]Observation, 200)
	for i := range obs {
		obs[i] = Observation{Duration: math.Ceil(rng.ExpFloat64() * 10), Event: rng.Float64() < 0.7}
	}
	var km KaplanMeier
	if err := km.Fit(obs); err != nil {
		t.Fatal(err)
	}
	band, err := km.Greenwood(0.95)
	if err != nil {
		t.Fatalf("Greenwood: %v", err)
	}
	if len(band.Times) != len(km.Curve().Times) {
		t.Fatalf("band has %d points, curve has %d", len(band.Times), len(km.Curve().Times))
	}
	for i := range band.Times {
		if band.Lo[i] < 0 || band.Hi[i] > 1 {
			t.Fatalf("band out of range at t=%v: [%v, %v]", band.Times[i], band.Lo[i], band.Hi[i])
		}
		if band.Lo[i] > band.Hi[i]+1e-12 {
			t.Fatalf("inverted band at t=%v", band.Times[i])
		}
		s := km.Curve().Survival[i]
		if s > 0 && s < 1 && (s < band.Lo[i]-1e-9 || s > band.Hi[i]+1e-9) {
			t.Fatalf("estimate %v outside its own band [%v, %v]", s, band.Lo[i], band.Hi[i])
		}
	}
}

func TestGreenwoodWidensWithLessData(t *testing.T) {
	mk := func(n int) ConfidenceBand {
		rng := rand.New(rand.NewSource(503))
		obs := make([]Observation, n)
		for i := range obs {
			obs[i] = Observation{Duration: math.Ceil(rng.ExpFloat64() * 5), Event: true}
		}
		var km KaplanMeier
		if err := km.Fit(obs); err != nil {
			t.Fatal(err)
		}
		b, err := km.Greenwood(0.95)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	small, large := mk(30), mk(3000)
	width := func(b ConfidenceBand) float64 {
		// Compare at the first time point both bands share.
		return b.Hi[0] - b.Lo[0]
	}
	if width(small) <= width(large) {
		t.Fatalf("less data should widen the band: %v vs %v", width(small), width(large))
	}
}

func TestGreenwoodValidation(t *testing.T) {
	var km KaplanMeier
	if _, err := km.Greenwood(0.95); !errors.Is(err, ErrNoData) {
		t.Fatalf("got %v, want ErrNoData", err)
	}
	if err := km.Fit([]Observation{{Duration: 1, Event: true}}); err != nil {
		t.Fatal(err)
	}
	for _, lvl := range []float64{0, 1, -0.5, 1.5} {
		if _, err := km.Greenwood(lvl); err == nil {
			t.Fatalf("level %v must be rejected", lvl)
		}
	}
}

func TestNelsonAalenMatchesKnownHazard(t *testing.T) {
	// Constant hazard 0.1 per unit time: H(t) = 0.1t.
	rng := rand.New(rand.NewSource(505))
	obs := make([]Observation, 20000)
	for i := range obs {
		obs[i] = Observation{Duration: rng.ExpFloat64() / 0.1, Event: true}
	}
	na, err := FitNelsonAalen(obs)
	if err != nil {
		t.Fatalf("FitNelsonAalen: %v", err)
	}
	for _, at := range []float64{5, 10, 20} {
		want := 0.1 * at
		got := na.At(at)
		if math.Abs(got-want) > 0.05*want+0.02 {
			t.Fatalf("H(%v) = %.4f, want ~%.4f", at, got, want)
		}
	}
}

func TestNelsonAalenStaysFiniteWhereKMHitsZero(t *testing.T) {
	// KM collapses to exactly 0 when the longest-observed subject fails.
	// The cumulative hazard keeps a readable slope, which is the practical
	// reason to have both.
	obs := []Observation{
		{Duration: 1, Event: true},
		{Duration: 2, Event: true},
		{Duration: 3, Event: true},
	}
	var km KaplanMeier
	if err := km.Fit(obs); err != nil {
		t.Fatal(err)
	}
	if km.Curve().At(3) != 0 {
		t.Fatalf("expected KM to reach 0, got %v", km.Curve().At(3))
	}
	na, err := FitNelsonAalen(obs)
	if err != nil {
		t.Fatal(err)
	}
	s := na.Survival()
	if s.At(3) <= 0 {
		t.Fatalf("Nelson-Aalen survival should stay above 0, got %v", s.At(3))
	}
	if na.At(0.5) != 0 {
		t.Fatalf("H before the first event should be 0, got %v", na.At(0.5))
	}
	for i := 1; i < len(na.Cumulative); i++ {
		if na.Cumulative[i] < na.Cumulative[i-1] {
			t.Fatal("cumulative hazard decreased")
		}
	}
}

func TestNelsonAalenValidation(t *testing.T) {
	if _, err := FitNelsonAalen(nil); !errors.Is(err, ErrNoData) {
		t.Fatalf("got %v, want ErrNoData", err)
	}
	if _, err := FitNelsonAalen([]Observation{{Duration: 1, Event: false}}); !errors.Is(err, ErrNoData) {
		t.Fatalf("all-censored: got %v, want ErrNoData", err)
	}
}

func TestLogRankDetectsRealDifference(t *testing.T) {
	rng := rand.New(rand.NewSource(507))
	mk := func(rate float64, n int) []Observation {
		out := make([]Observation, n)
		for i := range out {
			t0 := rng.ExpFloat64() / rate
			event := true
			if t0 > 30 {
				t0, event = 30, false
			}
			out[i] = Observation{Duration: math.Ceil(t0), Event: event}
		}
		return out
	}

	fast, slow := mk(0.2, 300), mk(0.05, 300)
	res, err := LogRank(fast, slow)
	if err != nil {
		t.Fatalf("LogRank: %v", err)
	}
	if !res.Significant(0.01) {
		t.Fatalf("a 4x hazard difference should be detected: %s", res)
	}
	// Group A failed faster, so it must show more failures than expected.
	if res.ObservedA <= res.ExpectedA {
		t.Fatalf("direction wrong: %s", res)
	}
	if res.Comparisons == 0 {
		t.Fatal("no event times contributed")
	}
}

func TestLogRankFindsNoDifferenceWhenThereIsNone(t *testing.T) {
	rng := rand.New(rand.NewSource(509))
	mk := func(n int) []Observation {
		out := make([]Observation, n)
		for i := range out {
			out[i] = Observation{Duration: math.Ceil(rng.ExpFloat64() / 0.1), Event: true}
		}
		return out
	}
	res, err := LogRank(mk(400), mk(400))
	if err != nil {
		t.Fatal(err)
	}
	if res.Significant(0.01) {
		t.Fatalf("identical distributions should not look different: %s", res)
	}
}

func TestLogRankPValueBounds(t *testing.T) {
	// A statistic of zero means no evidence at all, which is p=1.
	if p := chiSquareP1(0); p != 1 {
		t.Fatalf("chiSquareP1(0) = %v, want 1", p)
	}
	// Known landmark: chi-square 3.841 on 1 df is the 5% critical value.
	if p := chiSquareP1(3.841); math.Abs(p-0.05) > 0.001 {
		t.Fatalf("chiSquareP1(3.841) = %.4f, want ~0.05", p)
	}
	if p := chiSquareP1(6.635); math.Abs(p-0.01) > 0.001 {
		t.Fatalf("chiSquareP1(6.635) = %.4f, want ~0.01", p)
	}
}

func TestNormalQuantileKnownValues(t *testing.T) {
	tests := []struct{ p, want float64 }{
		{0.5, 0},
		{0.975, 1.959964},
		{0.95, 1.644854},
		{0.025, -1.959964},
		{0.005, -2.575829},
	}
	for _, tc := range tests {
		if got := normalQuantile(tc.p); math.Abs(got-tc.want) > 1e-4 {
			t.Fatalf("normalQuantile(%v) = %.6f, want %.6f", tc.p, got, tc.want)
		}
	}
}

func TestLogRankValidation(t *testing.T) {
	if _, err := LogRank(nil, []Observation{{Duration: 1, Event: true}}); !errors.Is(err, ErrNoData) {
		t.Fatalf("got %v, want ErrNoData", err)
	}
	allCensored := []Observation{{Duration: 1, Event: false}}
	if _, err := LogRank(allCensored, allCensored); !errors.Is(err, ErrNoData) {
		t.Fatalf("no events anywhere: got %v, want ErrNoData", err)
	}
}
