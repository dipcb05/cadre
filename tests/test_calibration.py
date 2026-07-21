from cadre.calibration import (
    CalibrationMethod,
    ProbabilityCalibrator,
    calibrate_complete_lineage_threshold,
)


def test_monotone_platt_predictions_are_bounded():
    calibrator = ProbabilityCalibrator(CalibrationMethod.MONOTONE_PLATT)
    calibrator.fit([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    values = calibrator.predict([0.15, 0.85])
    assert all(0.0 <= x <= 1.0 for x in values)
    assert values[0] <= values[1]


def test_lineage_threshold():
    result = calibrate_complete_lineage_threshold(
        probabilities=[0.9, 0.8, 0.7, 0.2, 0.6, 0.1],
        labels=[1, 1, 1, 0, 1, 0],
        lineage_ids=["a", "a", "b", "b", "c", "c"],
        grid=[0.0, 0.3, 0.5, 0.7],
        alpha=0.95,
        delta=0.05,
    )
    assert 0.0 <= result.threshold <= 1.0
