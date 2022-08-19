export CARLA_ROOT=/home/laura/DrivingSimulators/CARLA_0.9.10           # change to where you installed CARLA
export PORT=2000                                                    # change to port that CARLA is running on
# export ROUTES=./leaderboard/data/routes_training         # change to desired route
export TEAM_AGENT=auto_pilot.py                                    # no need to change
export TEAM_CONFIG=data/traffic_data/test                          # change path to checkpoint
export HAS_DISPLAY=1                                                # set to 0 if you don't want a debug window
export DEBUG_CHALLENGE=0

# ./collect_data.sh

search_dir=./leaderboard/data/routes_testing

for entry in "$search_dir"/*
do
  export ROUTES=$entry 
  ./collect_data.sh
done