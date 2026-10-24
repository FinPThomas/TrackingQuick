# TrackingQuick

Planning scaffold for the SPAICE technical challenge: visual detection and real-time 3D tracking of one flying drone. The initial proof of concept will be a 2D detector and tracker; the 3D system is a report design until it is implemented and measured.

Start with [OUTLINE.md](OUTLINE.md). It records the challenge requirements, proposed architecture, Kalman filter design, data plan, evaluation, and decisions to make before implementation.

## Planned layout

```text
TrackingQuick/
  OUTLINE.md            Challenge brief and proposed report outline
  README.md             Project entry point
  data/README.md        Dataset provenance and local data policy
  reports/              Final PDF and figures, when produced
  src/trackingquick/     Python implementation, when agreed
  tests/                 Focused checks for geometry and tracking
```

No dataset, model weights, training, or GPU setup is included at this stage.
