export CARLA_ROOT=/home/laura/DrivingSimulators/CARLA_0.9.10           # change to where you installed CARLA
export PORT=2000                                                    # change to port that CARLA is running on
export ROUTES=./leaderboard/data/routes_training.xml         # change to desired route
export TEAM_AGENT=traffic_dagger_agent.py                                   # no need to change
export TEAM_CONFIG=data/traffic_data/train                          # change path to checkpoint
export HAS_DISPLAY=0                                                # set to 0 if you don't want a debug window
export DEBUG_CHALLENGE=0

export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner

if [ -d "$TEAM_CONFIG" ]; then
    CHECKPOINT_ENDPOINT="$TEAM_CONFIG/$(basename $ROUTES .xml).txt"
else
    CHECKPOINT_ENDPOINT="$(dirname $TEAM_CONFIG)/$(basename $ROUTES .xml).txt"
fi

python train_dagger.py \
--track=SENSORS \
--scenarios=leaderboard/data/all_towns_traffic_scenarios_public.json  \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--routes=${ROUTES} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--carla-port=${PORT} \
--teacher_path=checkpoints/8-17_desired_vel/epoch\=17.ckpt \
--id=dagger_test \
--dataset_dir=${TEAM_CONFIG} \
--mode=train

echo "Done. See $CHECKPOINT_ENDPOINT for detailed results."