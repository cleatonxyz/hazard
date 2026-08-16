// Command horizon fits a discrete-time hazard model to synthetic subjects and
// reports how long each profile is expected to hold.
//
//	go run ./examples/horizon
package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/cleatonxyz/hazard"
)

func main() {
	rng := rand.New(rand.NewSource(1))

	// Two kinds of subject. The fragile one carries a covariate that raises its
	// per-period risk; a scheduled shock hits everyone after period 10.
	const nSubjects, maxPeriods = 3000, 20
	var rows []hazard.PersonPeriod
	for s := 0; s < nSubjects; s++ {
		fragile := 0.0
		if s%2 == 0 {
			fragile = 1
		}
		for p := 1; p <= maxPeriods; p++ {
			shock := 0.0
			if p > 10 {
				shock = 1
			}
			h := 0.02 + 0.06*fragile + 0.10*shock
			failed := rng.Float64() < h
			rows = append(rows, hazard.PersonPeriod{
				Period: p,
				X:      []float64{fragile, shock},
				Failed: failed,
			})
			if failed {
				break
			}
		}
	}

	d := &hazard.DiscreteTime{}
	if err := d.Fit(rows); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("coefficients (log-odds): fragile=%+.3f shock=%+.3f\n\n",
		d.Coefficients()[0], d.Coefficients()[1])

	profiles := []struct {
		name string
		x    []float64
	}{
		{"sturdy, no shock", []float64{0, 0}},
		{"fragile, no shock", []float64{1, 0}},
		{"fragile, shocked", []float64{1, 1}},
	}
	fmt.Printf("%-20s %10s %10s %10s\n", "profile", "S(10)", "hold@0.8", "hold@0.5")
	for _, p := range profiles {
		c, err := d.SurvivalConstant(p.x, maxPeriods)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%-20s %10.3f %10.0f %10.0f\n",
			p.name, c.At(10), c.HorizonAt(0.8), c.HorizonAt(0.5))
	}

	fmt.Println("\nhold@0.8 is the answer to \"how long is this safe\": the last period")
	fmt.Println("where survival is still at or above 80%.")
	fmt.Println("\nNote the modest shock coefficient: here the shock turns on for everyone")
	fmt.Println("at the same period, so the per-period baselines absorb most of its effect.")
	fmt.Println("A shock that hits different subjects on different dates separates cleanly.")
}
