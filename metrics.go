package hazard

import (
	"fmt"
	"math"
)

// Concordance is Harrell's C-index: the share of comparable subject pairs whose
// predicted risk ordering matches their observed ordering.
//
// 0.5 is coin-flipping, 1.0 is perfect ranking. A pair is comparable only when
// censoring does not make the comparison undecidable — if a censored subject
// left before the other failed, nobody knows who would have lasted longer, and
// counting that pair either way manufactures skill that was not demonstrated.
//
// Ties in risk count as half, which is what keeps a model that predicts one
// constant at 0.5 instead of accidentally scoring above it.
func Concordance(risk []float64, obs []Observation) (float64, error) {
	if len(risk) != len(obs) {
		return 0, fmt.Errorf("%w: %d risks vs %d observations", ErrShape, len(risk), len(obs))
	}
	if len(risk) == 0 {
		return 0, ErrNoData
	}
	var concordant, comparable float64
	for i := range obs {
		for j := i + 1; j < len(obs); j++ {
			a, b := obs[i], obs[j]
			var shorter, longer int
			switch {
			case a.Duration < b.Duration && a.Event:
				shorter, longer = i, j
			case b.Duration < a.Duration && b.Event:
				shorter, longer = j, i
			default:
				continue // not comparable
			}
			comparable++
			switch {
			case risk[shorter] > risk[longer]:
				concordant++
			case risk[shorter] == risk[longer]:
				concordant += 0.5
			}
		}
	}
	if comparable == 0 {
		return 0, fmt.Errorf("%w: no comparable pairs (all censored?)", ErrNoData)
	}
	return concordant / comparable, nil
}

// BrierScore measures squared error of predicted survival at one horizon,
// lower is better.
//
// Censored subjects still under observation at the horizon are excluded, since
// their outcome is unknown. This is the simple (uncensored-adjusted) form: it
// is unbiased when censoring is independent of risk, and biased when censoring
// is informative. If subjects leave the panel for reasons related to their risk,
// prefer IPCW weighting — which this package does not implement yet.
func BrierScore(predictedSurvival []float64, obs []Observation, horizon float64) (float64, error) {
	if len(predictedSurvival) != len(obs) {
		return 0, fmt.Errorf("%w: %d predictions vs %d observations", ErrShape, len(predictedSurvival), len(obs))
	}
	var sum, n float64
	for i, o := range obs {
		switch {
		case o.Duration <= horizon && o.Event:
			// Failed by the horizon: true survival is 0.
			sum += predictedSurvival[i] * predictedSurvival[i]
			n++
		case o.Duration > horizon:
			// Still alive at the horizon: true survival is 1.
			d := 1 - predictedSurvival[i]
			sum += d * d
			n++
		default:
			// Censored before the horizon: outcome unknown, excluded.
		}
	}
	if n == 0 {
		return 0, fmt.Errorf("%w: no subjects observable at horizon %v", ErrNoData, horizon)
	}
	return sum / n, nil
}

// CalibrationBin summarizes one bucket of a reliability check.
type CalibrationBin struct {
	Lo, Hi    float64 // predicted survival range
	N         int
	MeanPred  float64
	Empirical float64 // observed survival rate in the bin
}

// Gap is empirical minus predicted. Positive means the model was pessimistic.
func (b CalibrationBin) Gap() float64 { return b.Empirical - b.MeanPred }

// Calibration buckets predictions and compares each bucket's mean prediction
// with what actually happened.
//
// Discrimination and calibration are different properties: a model can rank
// subjects perfectly (C-index near 1) while every predicted probability is far
// from the truth. If the number gets published, this is the check that matters.
func Calibration(predictedSurvival []float64, obs []Observation, horizon float64, bins int) ([]CalibrationBin, error) {
	if len(predictedSurvival) != len(obs) {
		return nil, fmt.Errorf("%w: %d predictions vs %d observations", ErrShape, len(predictedSurvival), len(obs))
	}
	if bins < 1 {
		return nil, fmt.Errorf("hazard: bins must be >= 1, got %d", bins)
	}
	type acc struct {
		n, sumPred, survived float64
	}
	buckets := make([]acc, bins)
	for i, o := range obs {
		if o.Duration <= horizon && !o.Event {
			continue // censored before the horizon
		}
		p := predictedSurvival[i]
		idx := int(p * float64(bins))
		if idx >= bins {
			idx = bins - 1
		}
		if idx < 0 {
			idx = 0
		}
		buckets[idx].n++
		buckets[idx].sumPred += p
		if o.Duration > horizon {
			buckets[idx].survived++
		}
	}

	out := make([]CalibrationBin, 0, bins)
	width := 1 / float64(bins)
	for i, b := range buckets {
		if b.n == 0 {
			continue
		}
		out = append(out, CalibrationBin{
			Lo:        float64(i) * width,
			Hi:        float64(i+1) * width,
			N:         int(b.n),
			MeanPred:  b.sumPred / b.n,
			Empirical: b.survived / b.n,
		})
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%w: nothing observable at horizon %v", ErrNoData, horizon)
	}
	return out, nil
}

// MaxCalibrationGap is the largest absolute gap across bins holding at least
// minN subjects. Bins with a handful of subjects are noise, and letting them
// set the headline number makes a fine model look broken.
func MaxCalibrationGap(bins []CalibrationBin, minN int) float64 {
	worst := 0.0
	for _, b := range bins {
		if b.N < minN {
			continue
		}
		if g := math.Abs(b.Gap()); g > worst {
			worst = g
		}
	}
	return worst
}
