from cadre.calibration import CalibrationMethod
from cadre.risk import MonotoneRiskHead


fit_rows = [
    {"role_conflict": 0.0, "boundary_violation": 0.0, "script_anomaly": 0.0},
    {"role_conflict": 0.8, "boundary_violation": 1.0, "script_anomaly": 0.3},
    {"role_conflict": 0.1, "boundary_violation": 0.0, "script_anomaly": 0.1},
    {"role_conflict": 1.0, "boundary_violation": 1.0, "script_anomaly": 0.8},
]
fit_labels = [0, 1, 0, 1]

head = MonotoneRiskHead(
    ["role_conflict", "boundary_violation", "script_anomaly"],
    calibration=CalibrationMethod.MONOTONE_PLATT,
)
head.fit(fit_rows, fit_labels)
head.fit_calibrator(fit_rows, fit_labels)

print(head.estimate({"role_conflict": 0.9, "boundary_violation": 1.0, "script_anomaly": 0.2}))
