# Context-Aware Autonomous Security Patrol Drone

## Overview

This project implements a context-aware autonomous security patrol drone simulation using reinforcement learning, Gazebo, and risk-based decision making.

The drone patrols around a building, monitors a high-security entrance zone, observes an intruder approaching the restricted area, evaluates contextual risk, and switches behavior between patrol, investigation, and alert modes.

The project demonstrates how reinforcement learning can be combined with contextual risk assessment for autonomous security monitoring.

## Features

- Autonomous drone patrol around a building
- Gazebo simulation world with building, patrol area, and intruder marker
- PPO-based reinforcement learning navigation agent
- Context-aware risk assessment module
- High-security door/entrance monitoring
- Intruder approach and dwell-time simulation
- Dynamic behavior modes:
  - `PATROL`
  - `INVESTIGATE`
  - `ALERT`
- Risk scoring using:
  - restricted-zone proximity
  - dwell time
  - estimated detection confidence
  - distance to restricted zone
  - time-of-day factor
- Safe movement logic to avoid flying through the building

## Project Structure

```text
context_aware_drone/
├── README.md
├── README_RUN_STEPS.txt
├── requirements.txt
├── gazebo_rl_controller.py
├── risk_assessment.py
├── rl_navigation_env.py
├── train_ppo.py
├── patrol_world.sdf
└── ppo_drone_navigation.zip
