// Package optimize holds the gradient ascent used by the fitted models.
//
// It is internal because it is a means, not a feature: callers should judge the
// models by whether they recover known parameters, not by which optimizer got
// them there. Keeping it separate does mean the line-search logic gets tested
// on functions with known optima, instead of only through a statistical model
// where a bug looks like noise.
package optimize

import "math"

// Objective returns the value being maximized and its gradient at x.
type Objective func(x []float64) (value float64, gradient []float64)

// Options tune the ascent.
type Options struct {
	// MaxIter bounds the number of steps. Zero means 500.
	MaxIter int
	// Tol stops when the relative improvement falls below it. Zero means 1e-9.
	Tol float64
	// InitialStep is the first step size. Zero means 0.5.
	InitialStep float64
	// MaxBacktracks bounds the shrink loop per iteration. Zero means 40.
	MaxBacktracks int
}

// Result reports how the ascent ended.
type Result struct {
	X         []float64
	Value     float64
	Iters     int
	Converged bool
}

// Ascend maximizes obj starting from x0, using backtracking line search.
//
// Backtracking rather than a fixed learning rate: a fixed rate diverges as soon
// as one covariate is on a different scale from the others, and callers cannot
// be asked to guarantee otherwise. Newton's method would converge faster but
// needs a Hessian that is near-singular exactly when a period or a risk set is
// thin — which is normal in short panels, not exceptional.
func Ascend(obj Objective, x0 []float64, opts Options) Result {
	if opts.MaxIter == 0 {
		opts.MaxIter = 500
	}
	if opts.Tol == 0 {
		opts.Tol = 1e-9
	}
	if opts.InitialStep == 0 {
		opts.InitialStep = 0.5
	}
	if opts.MaxBacktracks == 0 {
		opts.MaxBacktracks = 40
	}

	x := append([]float64(nil), x0...)
	value, grad := obj(x)
	step := opts.InitialStep

	for iter := 1; iter <= opts.MaxIter; iter++ {
		improved := false
		for shrink := 0; shrink < opts.MaxBacktracks; shrink++ {
			candidate := make([]float64, len(x))
			for i := range x {
				candidate[i] = x[i] + step*grad[i]
			}
			newValue, newGrad := obj(candidate)
			if newValue > value {
				done := math.Abs(newValue-value) < opts.Tol*(math.Abs(value)+1)
				x, value, grad = candidate, newValue, newGrad
				step *= 1.1
				improved = true
				if done {
					return Result{X: x, Value: value, Iters: iter, Converged: true}
				}
				break
			}
			step /= 2
		}
		if !improved {
			// The step collapsed without finding an improvement: we are at an
			// optimum within machine precision, which is success rather than
			// failure.
			return Result{X: x, Value: value, Iters: iter, Converged: true}
		}
	}
	return Result{X: x, Value: value, Iters: opts.MaxIter, Converged: false}
}
