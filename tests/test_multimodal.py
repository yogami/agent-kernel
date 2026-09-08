"""Tests for Multimodal Kinetics & Biomechanical Safety Circuit Breakers."""

import pytest
from tools.multimodal_tools import (
    BiomechanicalAngleCalculatorTool,
    MovementSafetyCircuitBreakerTool,
    PoseLandmarkParserTool,
    VLMExerciseEvaluatorTool,
)


def test_pose_landmark_parser():
    """Verify landmark parsing, coordinate validation, and occlusion calculation."""
    tool = PoseLandmarkParserTool()
    sample_frames = [
        {
            "frame_index": 0,
            "timestamp_ms": 0.0,
            "landmarks": {
                "left_hip": {"x": 0.5, "y": 0.8, "z": 0.0, "visibility": 0.95},
                "left_knee": {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.90},
                "left_ankle": {"x": 0.5, "y": 0.1, "z": 0.0, "visibility": 0.85},
            },
        },
        {
            "frame_index": 1,
            "timestamp_ms": 33.3,
            "landmarks": {
                "left_hip": {"x": 0.5, "y": 0.78, "z": 0.0, "visibility": 0.95},
                "left_knee": {"x": 0.52, "y": 0.48, "z": 0.0, "visibility": 0.60},  # Low visibility
                "left_ankle": {"x": 0.5, "y": 0.1, "z": 0.0, "visibility": 0.85},
            },
        },
    ]

    result = tool.execute({"frames": sample_frames, "min_visibility": 0.7})
    assert result.is_error is False
    assert result.output["total_frames_parsed"] == 2
    assert result.output["quality_passed"] is True
    # 1 out of 6 landmarks is below 0.7 visibility -> occlusion rate ~ 0.167
    assert 0.15 < result.output["occlusion_rate"] < 0.20


def test_biomechanical_angle_calculator_geometric_truth():
    """Verify 3D angle math against known geometric angles (90 deg, 180 deg)."""
    tool = BiomechanicalAngleCalculatorTool()

    # 1. Test 90-degree right angle
    res_90 = tool.execute({
        "point_a": {"x": 0.0, "y": 1.0, "z": 0.0},
        "point_b": {"x": 0.0, "y": 0.0, "z": 0.0},
        "point_c": {"x": 1.0, "y": 0.0, "z": 0.0},
    })
    assert res_90.is_error is False
    assert pytest.approx(res_90.output["joint_angle_deg"], 0.01) == 90.0

    # 2. Test 180-degree straight leg
    res_180 = tool.execute({
        "point_a": {"x": 0.0, "y": 1.0, "z": 0.0},
        "point_b": {"x": 0.0, "y": 0.0, "z": 0.0},
        "point_c": {"x": 0.0, "y": -1.0, "z": 0.0},
    })
    assert pytest.approx(res_180.output["joint_angle_deg"], 0.01) == 180.0

    # 3. Test angular velocity
    res_vel = tool.execute({
        "point_a": {"x": 0.0, "y": 1.0, "z": 0.0},
        "point_b": {"x": 0.0, "y": 0.0, "z": 0.0},
        "point_c": {"x": 1.0, "y": 0.0, "z": 0.0},
        "previous_angle_deg": 120.0,
        "delta_time_sec": 0.1,  # 30 deg in 0.1s = 300 deg/s
    })
    assert pytest.approx(res_vel.output["angular_velocity_deg_per_sec"], 0.1) == 300.0


def test_movement_safety_circuit_breaker():
    """Verify deterministic safety rules intercept abnormal knee valgus and lumbar flexion."""
    tool = MovementSafetyCircuitBreakerTool()

    # Safe squat
    res_safe = tool.execute({
        "exercise_type": "squat",
        "knee_flexion_deg": 95.0,
        "knee_valgus_deg": 5.0,
        "lumbar_flexion_deg": 10.0,
    })
    assert res_safe.output["is_safe"] is True
    assert res_safe.output["circuit_breaker_tripped"] is False

    # Unsafe squat: severe knee valgus collapse (18 deg > 15 deg limit)
    res_valgus = tool.execute({
        "exercise_type": "squat",
        "knee_flexion_deg": 90.0,
        "knee_valgus_deg": 18.0,
    })
    assert res_valgus.output["is_safe"] is False
    assert res_valgus.output["circuit_breaker_tripped"] is True
    assert res_valgus.output["violations"][0]["rule"] == "KNEE_VALGUS_COLLAPSE"

    # Post-ACL patient with 10 deg valgus (exceeds stricter 8 deg limit)
    res_acl = tool.execute({
        "exercise_type": "squat",
        "knee_valgus_deg": 10.0,
        "patient_risk_profile": {"has_acl_injury": True},
    })
    assert res_acl.output["is_safe"] is False
    assert res_acl.output["violations"][0]["severity"] == "CRITICAL"


def test_vlm_exercise_evaluator():
    """Verify structured clinical exercise evaluation from rep kinematics."""
    tool = VLMExerciseEvaluatorTool()

    # 3 reps: 2 at ideal 90 deg depth, 1 shallow at 120 deg
    result = tool.execute({
        "exercise_name": "rehab_bodyweight_squat",
        "rep_minima_angles": [90.0, 92.0, 120.0],
        "safety_violations": [],
        "target_rom_min": 85.0,
        "target_rom_max": 105.0,
    })

    assert result.is_error is False
    output = result.output
    assert output["total_completed_reps"] == 3
    assert output["valid_target_depth_reps"] == 2
    assert output["shallow_reps"] == 1
    assert output["form_quality_score"] == 90.0
    assert output["clinical_status"] == "EXCELLENT"
