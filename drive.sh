export CARLA_ROOT=/home/laura/DrivingSimulators/CARLA_0.9.10           # change to where you installed CARLA
export PORT=3000                                                    # change to port that CARLA is running on
export ROUTES=./leaderboard/data/routes_devtest/route_00.xml         # change to desired route
export TEAM_AGENT=traffic_image_agent.py                                    # no need to change
export TEAM_CONFIG=checkpoints/toy_img_model/epoch=43.ckpt                                       # change path to checkpoint
export HAS_DISPLAY=1                                                # set to 0 if you don't want a debug window
export DEBUG_CHALLENGE=0

./run_agent.sh
