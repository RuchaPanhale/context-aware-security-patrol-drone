FRESH CONTEXT-AWARE DRONE PROJECT RUN STEPS

1. Make folder:
   mkdir -p ~/Desktop/context_aware_drone
   cd ~/Desktop/context_aware_drone

2. Put these Python files inside:
   risk_assessment.py
   rl_navigation_env.py
   train_ppo.py
   gazebo_rl_controller.py
   requirements.txt

3. Put patrol_world.sdf inside:
   ~/gazebo_world/patrol_world.sdf

4. Install dependencies:
   cd ~/Desktop/context_aware_drone
   python3 -m pip install -r requirements.txt

5. Train PPO:
   python3 train_ppo.py

6. Open Gazebo in Terminal 1:
   export GZ_IP=127.0.0.1
   export GZ_PARTITION=drone_project
   cd ~/gazebo_world
   gz sim patrol_world.sdf

7. Run controller in Terminal 2:
   export GZ_IP=127.0.0.1
   export GZ_PARTITION=drone_project
   cd ~/Desktop/context_aware_drone
   python3 gazebo_rl_controller.py

Expected:
- Drone moves around patrol area
- Intruder walks toward yellow door zone
- Risk changes LOW/MEDIUM/HIGH
- Drone changes PATROL/INVESTIGATE/ALERT
- Terminal prints YOLO confidence, dwell time, risk score, and mode
