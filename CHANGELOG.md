# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is [SemVer](https://semver.org/); while on `v0` the public API may
change in a minor release.

## [Unreleased]

### Added

- `Cox` proportional hazards with Efron (default) and Breslow tie handling,
  Breslow baseline hazard, `HazardRatios`, `RiskScore`, `Survival`, and
  `ProportionalityCheck` for the assumption the model rests on.
- `FitNelsonAalen`: cumulative hazard, which stays finite and readable where
  Kaplan-Meier collapses to zero.
- `KaplanMeier.Greenwood`: pointwise confidence bands on the log-log scale, so
  bounds cannot leave [0,1].
- `LogRank`: two-group comparison across the whole follow-up, reporting
  direction as well as a p-value.

### Changed

- Gradient ascent moved to `internal/optimize`, shared by `Cox` and
  `DiscreteTime` and tested against functions with known optima instead of only
  through a statistical model where a bug looks like noise.

## [0.1.0] - 2026-08-16

### Added

- Initial release.
