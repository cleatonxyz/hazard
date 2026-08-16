package optimize

import (
	"math"
	"testing"
)

// A quadratic with a known maximum: the cheapest way to tell a broken line
// search from a merely slow one.
func quadratic(center []float64) Objective {
	return func(x []float64) (float64, []float64) {
		v := 0.0
		g := make([]float64, len(x))
		for i := range x {
			d := x[i] - center[i]
			v -= d * d
			g[i] = -2 * d
		}
		return v, g
	}
}

func TestFindsKnownOptimum(t *testing.T) {
	center := []float64{3, -2, 0.5}
	res := Ascend(quadratic(center), make([]float64, 3), Options{})
	if !res.Converged {
		t.Fatalf("did not converge in %d iterations", res.Iters)
	}
	for i := range center {
		if math.Abs(res.X[i]-center[i]) > 1e-4 {
			t.Fatalf("x[%d] = %v, want %v", i, res.X[i], center[i])
		}
	}
	if math.Abs(res.Value) > 1e-7 {
		t.Fatalf("value = %v, want ~0", res.Value)
	}
}

func TestSurvivesBadlyScaledInputs(t *testing.T) {
	// One coordinate a million times larger than the others. A fixed learning
	// rate diverges here; backtracking must not.
	center := []float64{1e6, 1e-6}
	res := Ascend(quadratic(center), []float64{0, 0}, Options{MaxIter: 5000})
	if math.Abs(res.X[0]-center[0]) > 1 {
		t.Fatalf("x[0] = %v, want ~%v (diverged or stalled)", res.X[0], center[0])
	}
	if math.IsNaN(res.Value) || math.IsInf(res.Value, 0) {
		t.Fatalf("value blew up: %v", res.Value)
	}
}

func TestStartingAtTheOptimumIsConverged(t *testing.T) {
	center := []float64{1, 2}
	res := Ascend(quadratic(center), center, Options{})
	if !res.Converged {
		t.Fatal("starting at the optimum must report convergence")
	}
	if res.Iters > 2 {
		t.Fatalf("took %d iterations from the optimum", res.Iters)
	}
}

func TestRespectsIterationCap(t *testing.T) {
	// A ridge that improves forever but never converges within the cap.
	obj := func(x []float64) (float64, []float64) {
		return x[0], []float64{1}
	}
	res := Ascend(obj, []float64{0}, Options{MaxIter: 7, Tol: 1e-300})
	if res.Converged {
		t.Fatal("an unbounded objective must not report convergence")
	}
	if res.Iters != 7 {
		t.Fatalf("Iters = %d, want 7", res.Iters)
	}
}

func TestDoesNotMutateTheStartingPoint(t *testing.T) {
	x0 := []float64{0, 0}
	Ascend(quadratic([]float64{5, 5}), x0, Options{})
	if x0[0] != 0 || x0[1] != 0 {
		t.Fatalf("starting point was mutated: %v", x0)
	}
}
