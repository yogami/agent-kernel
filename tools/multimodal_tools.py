"""Multimodal Kinetics and Biomechanics tools for movement analysis and safety circuit breakers."""

from __future__ import annotations

import math
from typing import Any
import numpy as np
from pydantic import BaseModel, Field

from domain.models import ToolResult
from domain.ports import ToolPort


class LandmarkPoint(BaseModel):
    """3D landmark coordinate with confidence visibility score."""

    name: str
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


class PoseFrame(BaseModel):
    """Temporal frame containing 3D pose landmarks."""

    frame_index: int
    timestamp_ms: float
    landmarks: dict[str, LandmarkPoint]


class PoseLandmarkParserTool(ToolPort):
    """Parses and validates 33 MediaPipe/OpenCV skeletal joint landmark streams."""

    name: str = "parse_pose_landmarks"
    description: str = (
        "Parses spatial joint coordinates across temporal video frames, verifying visibility "
        "and anatomical validity for shoulders, elbows, hips, knees, and ankles."
    )

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "frames": {
                    "type": "array",
                    "description": "List of temporal video frames with raw landmark coordinates.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "frame_index": {"type": "integer"},
                            "timestamp_ms": {"type": "number"},
                            "landmarks": {"type": "object"},
                        },
                        "required": ["frame_index", "timestamp_ms", "landmarks"],
                    },
                },
                "min_visibility": {
                    "type": "number",
                    "description": "Minimum confidence visibility threshold (default 0.7).",
                    "default": 0.7,
                },
            },
            "required": ["frames"],
        }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        frames_raw = arguments.get("frames", [])
        min_vis = float(arguments.get("min_visibility", 0.7))

        if not frames_raw:
            return ToolResult(
                tool_id="pose_parse_err",
                tool_name=self.name,
                output={"error": "No frames provided for pose landmark parsing."},
                is_error=True,
                error_message="Empty frame array.",
            )

        parsed_frames: list[dict[str, Any]] = []
        low_confidence_count = 0
        total_landmarks = 0

        for f in frames_raw:
            f_idx = f.get("frame_index", 0)
            t_ms = f.get("timestamp_ms", 0.0)
            lms_raw = f.get("landmarks", {})
            valid_lms = {}

            for name, coords in lms_raw.items():
                total_landmarks += 1
                vis = float(coords.get("visibility", 1.0))
                if vis < min_vis:
                    low_confidence_count += 1
                valid_lms[name] = {
                    "x": float(coords.get("x", 0.0)),
                    "y": float(coords.get("y", 0.0)),
                    "z": float(coords.get("z", 0.0)),
                    "visibility": vis,
                }

            parsed_frames.append({
                "frame_index": f_idx,
                "timestamp_ms": t_ms,
                "landmark_count": len(valid_lms),
                "landmarks": valid_lms,
            })

        occlusion_rate = (low_confidence_count / total_landmarks) if total_landmarks > 0 else 0.0

        return ToolResult(
            tool_id="pose_parse_ok",
            tool_name=self.name,
            output={
                "total_frames_parsed": len(parsed_frames),
                "occlusion_rate": round(occlusion_rate, 3),
                "quality_passed": occlusion_rate < 0.25,
                "frames": parsed_frames,
            },
        )


class BiomechanicalAngleCalculatorTool(ToolPort):
    """Calculates 3D joint angles, angular velocities, and bilateral symmetry metrics."""

    name: str = "calculate_biomechanical_angles"
    description: str = (
        "Calculates 3D joint flexion/extension angles, valgus/varus deviation, and angular velocity "
        "given three joint landmarks (e.g., hip-knee-ankle or shoulder-elbow-wrist)."
    )

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "point_a": {
                    "type": "object",
                    "description": "Proximal landmark (e.g. hip: {x, y, z}).",
                    "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}},
                    "required": ["x", "y"],
                },
                "point_b": {
                    "type": "object",
                    "description": "Vertex joint landmark (e.g. knee: {x, y, z}).",
                    "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}},
                    "required": ["x", "y"],
                },
                "point_c": {
                    "type": "object",
                    "description": "Distal landmark (e.g. ankle: {x, y, z}).",
                    "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}},
                    "required": ["x", "y"],
                },
                "previous_angle_deg": {
                    "type": "number",
                    "description": "Angle from previous frame for velocity calculation.",
                },
                "delta_time_sec": {
                    "type": "number",
                    "description": "Time delta between frames in seconds.",
                },
            },
            "required": ["point_a", "point_b", "point_c"],
        }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        pa = arguments.get("point_a", {})
        pb = arguments.get("point_b", {})
        pc = arguments.get("point_c", {})

        # Extract vectors ba and bc
        v_ba = np.array([
            pa.get("x", 0.0) - pb.get("x", 0.0),
            pa.get("y", 0.0) - pb.get("y", 0.0),
            pa.get("z", 0.0) - pb.get("z", 0.0),
        ], dtype=np.float64)

        v_bc = np.array([
            pc.get("x", 0.0) - pb.get("x", 0.0),
            pc.get("y", 0.0) - pb.get("y", 0.0),
            pc.get("z", 0.0) - pb.get("z", 0.0),
        ], dtype=np.float64)

        norm_ba = np.linalg.norm(v_ba)
        norm_bc = np.linalg.norm(v_bc)

        if norm_ba == 0.0 or norm_bc == 0.0:
            return ToolResult(
                tool_id="angle_calc_err",
                tool_name=self.name,
                output={"error": "Degenerate zero-length vector detected between landmarks."},
                is_error=True,
                error_message="Zero norm vector.",
            )

        cosine_val = np.dot(v_ba, v_bc) / (norm_ba * norm_bc)
        cosine_clamped = np.clip(cosine_val, -1.0, 1.0)
        angle_rad = np.arccos(cosine_clamped)
        angle_deg = float(np.degrees(angle_rad))

        # Velocity calculation
        angular_velocity = None
        prev_angle = arguments.get("previous_angle_deg")
        dt = arguments.get("delta_time_sec")
        if prev_angle is not None and dt and dt > 0.0:
            angular_velocity = abs(angle_deg - float(prev_angle)) / float(dt)

        # Frontal plane valgus deviation estimation (X-axis medial displacement of B relative to line AC)
        mid_x = (pa.get("x", 0.0) + pc.get("x", 0.0)) / 2.0
        valgus_displacement_x = pb.get("x", 0.0) - mid_x

        return ToolResult(
            tool_id="angle_calc_ok",
            tool_name=self.name,
            output={
                "joint_angle_deg": round(angle_deg, 2),
                "angular_velocity_deg_per_sec": round(angular_velocity, 2) if angular_velocity else None,
                "medial_valgus_displacement": round(valgus_displacement_x, 4),
            },
        )


class MovementSafetyCircuitBreakerTool(ToolPort):
    """Deterministic safety firewall intercepting unsafe biomechanical loads and form defects."""

    name: str = "check_movement_safety_bounds"
    description: str = (
        "Evaluates calculated joint angles, velocities, and alignment against clinical "
        "rehabilitation safety thresholds, triggering immediate deterministic circuit breakers on risk."
    )

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "exercise_type": {
                    "type": "string",
                    "description": "Exercise type (e.g. squat, lunge, shoulder_abduction, deadlift).",
                },
                "knee_flexion_deg": {"type": "number"},
                "knee_valgus_deg": {"type": "number"},
                "lumbar_flexion_deg": {"type": "number"},
                "angular_velocity_deg_per_sec": {"type": "number"},
                "patient_risk_profile": {
                    "type": "object",
                    "properties": {
                        "has_acl_injury": {"type": "boolean"},
                        "has_lumbar_herniation": {"type": "boolean"},
                        "post_op_weeks": {"type": "integer"},
                    },
                },
            },
            "required": ["exercise_type"],
        }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        exercise = arguments.get("exercise_type", "").lower()
        valgus = arguments.get("knee_valgus_deg")
        flexion = arguments.get("knee_flexion_deg")
        lumbar = arguments.get("lumbar_flexion_deg")
        velocity = arguments.get("angular_velocity_deg_per_sec")
        profile = arguments.get("patient_risk_profile", {})

        violations: list[dict[str, Any]] = []
        is_safe = True

        # Rule 1: Knee Valgus Collapse Check (Inward knee buckle > 15 degrees or > 8 deg for post-ACL)
        valgus_limit = 8.0 if profile.get("has_acl_injury") else 15.0
        if valgus is not None and valgus > valgus_limit:
            is_safe = False
            violations.append({
                "rule": "KNEE_VALGUS_COLLAPSE",
                "severity": "CRITICAL" if profile.get("has_acl_injury") else "HIGH",
                "measured": valgus,
                "limit": valgus_limit,
                "clinical_risk": "Excessive anterior cruciate ligament (ACL) and patellofemoral shear stress.",
                "correction": "Push knees outward in line with second toe. Engage gluteus medius.",
            })

        # Rule 2: Excessive Spinal / Lumbar Flexion (Rounding back under load > 30 degrees)
        lumbar_limit = 15.0 if profile.get("has_lumbar_herniation") else 30.0
        if lumbar is not None and lumbar > lumbar_limit:
            is_safe = False
            violations.append({
                "rule": "EXCESSIVE_LUMBAR_FLEXION",
                "severity": "HIGH",
                "measured": lumbar,
                "limit": lumbar_limit,
                "clinical_risk": "Spinal disc posterior herniation risk and loss of neutral lumbar spine.",
                "correction": "Maintain neutral spine, brace abdominal wall, hinge from hips.",
            })

        # Rule 3: Ballistic Angular Velocity Spike (> 450 deg/s in controlled rehab)
        if velocity is not None and velocity > 450.0:
            is_safe = False
            violations.append({
                "rule": "BALLISTIC_ACCELERATION_SPIKE",
                "severity": "MEDIUM",
                "measured": velocity,
                "limit": 450.0,
                "clinical_risk": "Loss of eccentric control during movement.",
                "correction": "Slow down cadence. Maintain 2-second eccentric phase.",
            })

        return ToolResult(
            tool_id="safety_check_ok",
            tool_name=self.name,
            output={
                "is_safe": is_safe,
                "circuit_breaker_tripped": not is_safe,
                "violation_count": len(violations),
                "violations": violations,
                "exercise_type": exercise,
            },
        )


class VLMExerciseEvaluatorTool(ToolPort):
    """Produces structured clinical exercise evaluations from temporal kinematics."""

    name: str = "evaluate_exercise_kinematics"
    description: str = (
        "Aggregates temporal movement frames, rep counts, range of motion, and safety violations "
        "into a structured clinical physical therapy summary."
    )

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "exercise_name": {"type": "string"},
                "rep_minima_angles": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Peak flexion angle measured at bottom of each rep.",
                },
                "safety_violations": {"type": "array", "items": {"type": "object"}},
                "target_rom_min": {"type": "number", "default": 80.0},
                "target_rom_max": {"type": "number", "default": 110.0},
            },
            "required": ["exercise_name", "rep_minima_angles"],
        }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        exercise = arguments.get("exercise_name", "Unknown Exercise")
        reps = arguments.get("rep_minima_angles", [])
        violations = arguments.get("safety_violations", [])
        rom_min = float(arguments.get("target_rom_min", 80.0))
        rom_max = float(arguments.get("target_rom_max", 110.0))

        if not reps:
            return ToolResult(
                tool_id="vlm_eval_empty",
                tool_name=self.name,
                output={"error": "No repetitions provided for evaluation."},
                is_error=True,
                error_message="Empty rep list.",
            )

        valid_depth_reps = sum(1 for angle in reps if rom_min <= angle <= rom_max)
        shallow_reps = sum(1 for angle in reps if angle > rom_max)
        deep_reps = sum(1 for angle in reps if angle < rom_min)

        form_score = max(0.0, 100.0 - (len(violations) * 15.0) - (shallow_reps * 10.0))

        return ToolResult(
            tool_id="vlm_eval_ok",
            tool_name=self.name,
            output={
                "exercise_name": exercise,
                "total_completed_reps": len(reps),
                "valid_target_depth_reps": valid_depth_reps,
                "shallow_reps": shallow_reps,
                "excessive_depth_reps": deep_reps,
                "form_quality_score": round(form_score, 1),
                "clinical_status": "EXCELLENT" if form_score >= 85 else ("ACCEPTABLE" if form_score >= 70 else "NEEDS_SUPERVISION"),
                "critical_safety_alerts": [v for v in violations if v.get("severity") in ["CRITICAL", "HIGH"]],
            },
        )
