package hazard

import (
	"fmt"
	"math"
	"sort"
)

// ConfidenceBand is a pointwise interval around a survival curve.
//
// Pointwise, not simultaneous: each time point's interval covers the truth at
// the stated level, but the probability that the *whole curve* stays inside the
// band is lower. Quoting a pointwise band as if it covered the entire curve is
// the standard way to overstate certainty about a survival estimate.
type ConfidenceBand struct {
	Times []float64
	Lo    []float64
	Hi    []float64
	Level float64
}

// Greenwood returns a pointwise confidence band for a Kaplan-Meier curve.
//
// The interval is built on the log-log scale rather than directly on S(t). A
// naive S ± z·se runs past 1 near the start and below 0 in the tail, which is
// not merely ugly — it is a band that includes impossible values, and readers
// reasonably conclude the estimate is broken.
func (km *KaplanMeier) Greenwood(level float64) (ConfidenceBand, error) {
	if len(km.curve.Times) == 0 {
		return ConfidenceBand{}, ErrNoData
	}
	if level <= 0 || level >= 1 {
		return ConfidenceBand{}, fmt.Errorf("hazard: level must be in (0,1), got %v", level)
	}
	z := normalQuantile(1 - (1-level)/2)

	band := ConfidenceBand{Level: level}
	cumVar := 0.0
	for i, t := range km.curve.Times {
		d := km.curve.Events[i]
		n := km.curve.AtRisk[i]
		s := km.curve.Survival[i]

		if n > d && n > 0 {
			cumVar += d / (n * (n - d))
		}
		band.Times = append(band.Times, t)

		if s <= 0 || s >= 1 || cumVar <= 0 {
			// At S=0 or S=1 the log-log transform is undefined; the honest
			// answer is a degenerate interval rather than a fabricated one.
			band.Lo = append(band.Lo, s)
			band.Hi = append(band.Hi, s)
			continue
		}
		logS := math.Log(s)
		se := math.Sqrt(cumVar) / math.Abs(logS)
		theta := math.Exp(z * se)
		band.Lo = append(band.Lo, math.Pow(s, theta))
		band.Hi = append(band.Hi, math.Pow(s, 1/theta))
	}
	return band, nil
}

// NelsonAalen estimates the cumulative hazard H(t) directly.
//
// Where Kaplan-Meier multiplies survival probabilities, Nelson-Aalen adds
// hazards. The practical difference shows up in small risk sets: KM collapses
// to exactly zero the moment the last observed subject fails, while the
// cumulative hazard stays finite and keeps its slope readable. When the
// question is "is risk accelerating", the hazard is the curve to look at.
type NelsonAalen struct {
	Times      []float64
	Cumulative []float64
	Variance   []float64
}

// FitNelsonAalen computes the cumulative hazard estimate.
func FitNelsonAalen(obs []Observation) (*NelsonAalen, error) {
	if len(obs) == 0 {
		return nil, ErrNoData
	}
	sorted := append([]Observation(nil), obs...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Duration < sorted[j].Duration })

	atRisk := 0.0
	for _, o := range sorted {
		atRisk += o.weight()
	}

	na := &NelsonAalen{}
	cum, variance := 0.0, 0.0
	i := 0
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
		if events > 0 && atRisk > 0 {
			cum += events / atRisk
			variance += events / (atRisk * atRisk)
			na.Times = append(na.Times, t)
			na.Cumulative = append(na.Cumulative, cum)
			na.Variance = append(na.Variance, variance)
		}
		atRisk -= events + censored
	}
	if len(na.Times) == 0 {
		return nil, fmt.Errorf("%w: no events observed", ErrNoData)
	}
	return na, nil
}

// At returns H(t), carried forward from the last event time at or before t.
func (na *NelsonAalen) At(t float64) float64 {
	if len(na.Times) == 0 || t < na.Times[0] {
		return 0
	}
	i := sort.SearchFloat64s(na.Times, t)
	if i == len(na.Times) || na.Times[i] > t {
		i--
	}
	return na.Cumulative[i]
}

// Survival converts the cumulative hazard to a survival curve via
// S(t) = exp(-H(t)).
//
// It differs slightly from Kaplan-Meier and neither is wrong; the exponential
// form is smoother in small samples and never reaches exactly zero.
func (na *NelsonAalen) Survival() Curve {
	c := Curve{Times: append([]float64(nil), na.Times...)}
	c.Survival = make([]float64, len(na.Cumulative))
	for i, h := range na.Cumulative {
		c.Survival[i] = math.Exp(-h)
	}
	return c
}

// LogRankResult reports whether two groups' survival differs.
type LogRankResult struct {
	// Statistic is the chi-square statistic on one degree of freedom.
	Statistic float64
	// PValue is the two-sided p-value.
	PValue float64
	// ObservedA and ExpectedA are the failure counts in group A. Comparing them
	// says which direction the difference runs, which the statistic alone does
	// not.
	ObservedA, ExpectedA float64
	// Comparisons is the number of event times that contributed.
	Comparisons int
}

// Significant reports whether the p-value is below alpha.
func (r LogRankResult) Significant(alpha float64) bool { return r.PValue < alpha }

func (r LogRankResult) String() string {
	dir := "worse"
	if r.ObservedA < r.ExpectedA {
		dir = "better"
	}
	return fmt.Sprintf("chi2=%.3f p=%.4f (group A did %s than expected: %.1f observed vs %.1f)",
		r.Statistic, r.PValue, dir, r.ObservedA, r.ExpectedA)
}

// LogRank tests whether two groups have the same survival function.
//
// It compares observed failures against what would be expected if the groups
// were identical, pooling across every event time. Unlike comparing two medians
// or two curves at a fixed date, it uses the whole follow-up and handles
// censoring — which is why it is the standard test here.
//
// Note what it is sensitive to: the log-rank test is most powerful when hazards
// are proportional. Two curves that cross can produce a large p-value while
// being obviously different, because early and late differences cancel.
func LogRank(groupA, groupB []Observation) (LogRankResult, error) {
	if len(groupA) == 0 || len(groupB) == 0 {
		return LogRankResult{}, ErrNoData
	}

	times := map[float64]struct{}{}
	for _, o := range append(append([]Observation{}, groupA...), groupB...) {
		if o.Event {
			times[o.Duration] = struct{}{}
		}
	}
	eventTimes := make([]float64, 0, len(times))
	for t := range times {
		eventTimes = append(eventTimes, t)
	}
	sort.Float64s(eventTimes)
	if len(eventTimes) == 0 {
		return LogRankResult{}, fmt.Errorf("%w: no events in either group", ErrNoData)
	}

	var observedA, expectedA, variance float64
	comparisons := 0
	for _, t := range eventTimes {
		nA, dA := atRiskAndEvents(groupA, t)
		nB, dB := atRiskAndEvents(groupB, t)
		n, d := nA+nB, dA+dB
		if n <= 1 || d == 0 {
			continue
		}
		observedA += dA
		expectedA += d * nA / n
		// Hypergeometric variance of dA given the margins.
		variance += d * (nA / n) * (nB / n) * (n - d) / (n - 1)
		comparisons++
	}
	if variance <= 0 {
		return LogRankResult{}, fmt.Errorf("%w: no comparable event times", ErrNoData)
	}

	diff := observedA - expectedA
	stat := diff * diff / variance
	return LogRankResult{
		Statistic:   stat,
		PValue:      chiSquareP1(stat),
		ObservedA:   observedA,
		ExpectedA:   expectedA,
		Comparisons: comparisons,
	}, nil
}

func atRiskAndEvents(obs []Observation, t float64) (atRisk, events float64) {
	for _, o := range obs {
		if o.Duration >= t {
			atRisk += o.weight()
		}
		if o.Duration == t && o.Event {
			events += o.weight()
		}
	}
	return
}

// chiSquareP1 is the upper tail of a chi-square with one degree of freedom,
// which is erfc(sqrt(x/2)).
func chiSquareP1(x float64) float64 {
	if x <= 0 {
		return 1
	}
	return math.Erfc(math.Sqrt(x / 2))
}

// normalQuantile is the inverse standard normal CDF, via the Acklam
// approximation. Accurate to about 1e-9, which is far beyond what a survival
// confidence band needs.
func normalQuantile(p float64) float64 {
	if p <= 0 {
		return math.Inf(-1)
	}
	if p >= 1 {
		return math.Inf(1)
	}
	a := []float64{-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
		1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00}
	b := []float64{-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
		6.680131188771972e+01, -1.328068155288572e+01}
	c := []float64{-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
		-2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00}
	d := []float64{7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
		3.754408661907416e+00}

	const pLow, pHigh = 0.02425, 1 - 0.02425
	switch {
	case p < pLow:
		q := math.Sqrt(-2 * math.Log(p))
		return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q + c[5]) /
			((((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1)
	case p <= pHigh:
		q := p - 0.5
		r := q * q
		return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r + a[5]) * q /
			(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r + 1)
	default:
		q := math.Sqrt(-2 * math.Log(1-p))
		return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q + c[5]) /
			((((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1)
	}
}
