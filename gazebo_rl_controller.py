import math
import subprocess
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from risk_assessment import RiskAssessor


DRONE_NAME = "drone"
INTRUDER_NAME = "intruder_marker"

DRONE_Z = 1.6
INTRUDER_Z = 0.08

MODEL_PATH = "ppo_drone_navigation.zip"

# Patrol route around building
PATROL_POINTS = [
    (-4.5, 4.5),
    (4.5, 4.5),
    (4.5, -4.5),
    (-4.5, -4.5),
]

# One high-security entrance zone in front of building door
HIGH_SECURITY_DOOR = (0.0, -2.8)

# Intruder approaches the building entrance, lingers, then leaves
INTRUDER_WAYPOINTS = [
    (0, (-5.5, -4.8)),
    (5, (-3.0, -4.0)),
    (10, (-1.2, -3.2)),
    (15, (0.0, -2.8)),
    (22, (0.0, -2.8)),
    (28, (2.8, -4.2)),
    (35, (5.5, -4.8)),
]


def run_cmd(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def detect_world_name():
    result = run_cmd(["gz", "service", "-l"])
    lines = result.stdout.splitlines()

    for line in lines:
        if line.endswith("/set_pose") and line.startswith("/world/"):
            parts = line.split("/")
            if len(parts) >= 3:
                return parts[2]

    # Common fallback on macOS Gazebo
    return "default"


def set_model_pose(world_name, model_name, x, y, z, yaw=0.0):
    req = (
        f'name: "{model_name}" '
        f'position {{ x: {x:.3f} y: {y:.3f} z: {z:.3f} }} '
        f'orientation {{ x: 0 y: 0 z: {math.sin(yaw / 2):.5f} w: {math.cos(yaw / 2):.5f} }}'
    )

    cmd = [
        "gz",
        "service",
        "-s",
        f"/world/{world_name}/set_pose",
        "--reqtype",
        "gz.msgs.Pose",
        "--reptype",
        "gz.msgs.Boolean",
        "--timeout",
        "1000",
        "--req",
        req,
    ]

    result = run_cmd(cmd)
    return result.returncode == 0


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def get_intruder_position(elapsed):
    cycle_time = INTRUDER_WAYPOINTS[-1][0]
    t = elapsed % cycle_time

    for i in range(len(INTRUDER_WAYPOINTS) - 1):
        t1, p1 = INTRUDER_WAYPOINTS[i]
        t2, p2 = INTRUDER_WAYPOINTS[i + 1]

        if t1 <= t <= t2:
            ratio = (t - t1) / (t2 - t1)
            x = p1[0] + (p2[0] - p1[0]) * ratio
            y = p1[1] + (p2[1] - p1[1]) * ratio
            return (x, y)

    return INTRUDER_WAYPOINTS[0][1]


def choose_target(risk_level, patrol_index, intruder_pos):
    if risk_level == "HIGH":
        return intruder_pos, "ALERT"
    if risk_level == "MEDIUM":
        return intruder_pos, "INVESTIGATE"
    return PATROL_POINTS[patrol_index], "PATROL"


def get_obs(drone_pos, target_pos):
    dx = target_pos[0] - drone_pos[0]
    dy = target_pos[1] - drone_pos[1]
    return np.array(
        [drone_pos[0], drone_pos[1], target_pos[0], target_pos[1], dx, dy],
        dtype=np.float32,
    )


def action_to_motion(action, step=0.35):
    if action == 0:
        return (0.0, step)
    if action == 1:
        return (0.0, -step)
    if action == 2:
        return (-step, 0.0)
    if action == 3:
        return (step, 0.0)
    return (0.0, 0.0)


def inside_building(pos):
    x, y = pos
    return -1.4 <= x <= 1.4 and -1.2 <= y <= 1.2


def safe_move(drone_pos, action):
    dx, dy = action_to_motion(action)
    new_pos = (drone_pos[0] + dx, drone_pos[1] + dy)

    # Keep inside world limits
    new_pos = (
        max(-6.0, min(6.0, new_pos[0])),
        max(-6.0, min(6.0, new_pos[1])),
    )

    # Prevent flying through building
    if inside_building(new_pos):
        return drone_pos

    return new_pos


def heuristic_action(drone_pos, target_pos):
    """
    Backup controller only if PPO model is missing.
    The main demo should use PPO after train_ppo.py is run.
    """
    dx = target_pos[0] - drone_pos[0]
    dy = target_pos[1] - drone_pos[1]

    if abs(dx) > abs(dy):
        return 3 if dx > 0 else 2
    return 0 if dy > 0 else 1


def main():
    print("Starting context-aware RL Gazebo controller...")

    world_name = detect_world_name()
    print(f"Detected Gazebo world: {world_name}")

    model = None
    if Path(MODEL_PATH).exists():
        model = PPO.load(MODEL_PATH)
        print("Loaded PPO model.")
    else:
        print("WARNING: PPO model not found. Run python3 train_ppo.py first.")
        print("Using backup navigation only so Gazebo demo can still move.")

    risk_assessor = RiskAssessor(restricted_zone=HIGH_SECURITY_DOOR)

    drone_pos = PATROL_POINTS[0]
    patrol_index = 1

    set_model_pose(world_name, DRONE_NAME, drone_pos[0], drone_pos[1], DRONE_Z)
    set_model_pose(world_name, INTRUDER_NAME, -5.5, -4.8, INTRUDER_Z)

    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        intruder_pos = get_intruder_position(elapsed)

        # Update intruder in Gazebo
        set_model_pose(world_name, INTRUDER_NAME, intruder_pos[0], intruder_pos[1], INTRUDER_Z)

        # Contextual risk assessment
        risk = risk_assessor.assess(intruder_pos, is_night=False)
        target, mode = choose_target(risk["level"], patrol_index, intruder_pos)

        # PPO action selection
        obs = get_obs(drone_pos, target)

        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = heuristic_action(drone_pos, target)

        old_pos = drone_pos
        drone_pos = safe_move(drone_pos, action)

        # Update patrol checkpoint if in patrol mode
        if mode == "PATROL" and distance(drone_pos, target) < 0.6:
            patrol_index = (patrol_index + 1) % len(PATROL_POINTS)

        yaw = math.atan2(drone_pos[1] - old_pos[1], drone_pos[0] - old_pos[0])
        set_model_pose(world_name, DRONE_NAME, drone_pos[0], drone_pos[1], DRONE_Z, yaw)

        print(
            f"Mode={mode:11s} "
            f"Risk={risk['level']:6s} "
            f"Score={risk['score']:.2f} "
            f"YOLOconf={risk['yolo_confidence']:.2f} "
            f"Dwell={risk['dwell_time']:.1f}s "
            f"Drone=({drone_pos[0]:.2f},{drone_pos[1]:.2f}) "
            f"Intruder=({intruder_pos[0]:.2f},{intruder_pos[1]:.2f}) "
            f"Target=({target[0]:.2f},{target[1]:.2f})"
        )

        time.sleep(0.18)


if __name__ == "__main__":
    main()
