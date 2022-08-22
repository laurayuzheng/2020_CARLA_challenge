import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
from distutils.version import LooseVersion
import importlib
import os
import pkg_resources
import sys
import torchvision
import gym
from gym import spaces
import numpy as np

import carla
import signal
import sumolib
import pathlib

sys.path.append("./")
sys.path.append("./leaderboard")
sys.path.append("./scenario_runner")

import lxml.etree as ET  # pylint: disable=wrong-import-position
from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper import  AgentWrapper, AgentError
from leaderboard.utils.route_indexer import RouteIndexer

# from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from carla_project.src.run_synchronization import SimulationSynchronization
from carla_project.src.traffic.rewards import * 
from sumo_integration.bridge_helper import BridgeHelper  # pylint: disable=wrong-import-position
from sumo_integration.carla_simulation import CarlaSimulation  # pylint: disable=wrong-import-position
from sumo_integration.constants import INVALID_ACTOR_ID  # pylint: disable=wrong-import-position
from sumo_integration.sumo_simulation import SumoSimulation  # pylint: disable=wrong-import-position
from sumo_integration.util.netconvert_carla import netconvert_carla

from team_code.accel_agent import AccelAgent

PROJECT_ROOT = "/scratch/2020_CARLA_challenge"
CARLA_HOME = "/home/laura/DrivingSimulators/CARLA_0.9.10"

sensors_to_icons = {
    'sensor.camera.semantic_segmentation':        'carla_camera',
    'sensor.camera.rgb':        'carla_camera',
    'sensor.lidar.ray_cast':    'carla_lidar',
    'sensor.other.radar':       'carla_radar',
    'sensor.other.gnss':        'carla_gnss',
    'sensor.other.imu':         'carla_imu',
    'sensor.opendrive_map':     'carla_opendrive_map',
    'sensor.speedometer':       'carla_speedometer'
}

def write_sumocfg_xml(cfg_file, net_file, vtypes_file, viewsettings_file, additional_traci_clients=0):
    """
    Writes sumo configuration xml file.
    """
    root = ET.Element('configuration')

    input_tag = ET.SubElement(root, 'input')
    ET.SubElement(input_tag, 'net-file', {'value': net_file})
    ET.SubElement(input_tag, 'route-files', {'value': vtypes_file})

    gui_tag = ET.SubElement(root, 'gui_only')
    ET.SubElement(gui_tag, 'gui-settings-file', {'value': viewsettings_file})

    ET.SubElement(root, 'num-clients', {'value': str(additional_traci_clients+1)})

    tree = ET.ElementTree(root)

    # with open(cfg_file, 'w+') as f:
    tree.write(cfg_file, pretty_print=True, encoding='UTF-8', xml_declaration=True)

HEIGHT = 88
WIDTH = 200 
N_CHANNELS = 3


class CosimEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, args):
        super(CosimEnv, self).__init__()
        self.args = args
        self.cosim_manager = CosimManager(args)
        self.num_vehicles = 5

    @property
    def action_space(self):
        """See class definition."""
        return spaces.Box(
            low=-3,
            high=3,
            shape=(1, ), # One action, one vehicle
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        self.obs_var_labels = ['Velocity', 'Absolute_pos']
        return spaces.Box(low=0, high=np.inf, # position can be infinitely large
                                shape=(2 * self.num_vehicles, ),
                                dtype=np.float32)

        # spaces_dict = {
        #     # Defines image observation
        #     'rgb': spaces.Box(low=0, high=255, 
        #                         shape=(HEIGHT, WIDTH, N_CHANNELS), 
        #                         dtype=np.uint8),
            
        #     # Defines state observation
        #     'traffic_state': spaces.Box(low=0, high=np.inf, # position can be infinitely large
        #                         shape=(2 * self.num_vehicles, ),
        #                         dtype=np.float32)
        # }
        # return gym.spaces.Dict(spaces_dict)

    def step(self, action):
        action = np.squeeze(action)
        rgb, traffic_state, player_ind, done = self.cosim_manager.tick_scenario(action)
        
        # observation = {
        #     'rgb': rgb, 
        #     'traffic_state': traffic_state
        # }

        reward = self.d_compute_reward(traffic_state)
        
        info = {}
        return traffic_state, reward, done, info
    
    def d_compute_reward(self, state):
        if self.args.reward_type == "avg_velocity":
            return d_average_velocity(state, fail=False)
        else: 
            return d_desired_velocity(state, 30, fail=False)

    def reset(self):
        self.cosim_manager.start_single_episode(restart=True)
        results = self.cosim_manager.tick_scenario(0.)
        
        if results is not None:
            rgb, traffic_state, player_ind, done = results
            self.num_vehicles = len(traffic_state) // 2
            return traffic_state 
        else: 
            return np.zeros((10,))

        # return {
        #     'rgb': rgb, 
        #     'traffic_state': state
        # }  

    def render(self, mode='human'):
        pass

    def close (self):
        self.cosim_manager.synchronization.close() 
        self.cosim_manager = CosimManager(self.args)


class CosimManager(object):
    ego_vehicles = []

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    wait_for_world = 20.0  # in seconds
    frame_rate = 20.0      # in Hz

    def __init__(self, args):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        self.sensors = None
        self.sensor_icons = []
        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam

        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        self.carla_simulation = CarlaSimulation(args.carla_host, args.carla_port, 1/20)
        self.client = self.carla_simulation.client

        if args.timeout:
            self.client_timeout = float(args.timeout)
        self.client.set_timeout(self.client_timeout)

        self.traffic_manager = self.client.get_trafficmanager(int(args.trafficManagerPort))

        dist = pkg_resources.get_distribution("carla")
        if dist.version != 'leaderboard':
            if LooseVersion(dist.version) < LooseVersion('0.9.10'):
                raise ImportError("CARLA version 0.9.10.1 or newer required. CARLA version found: {}".format(dist))

        # Load agent
        module_name = os.path.basename(args.agent).split('.')[0]
        sys.path.insert(0, os.path.dirname(args.agent))
        self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        self.manager = ScenarioManager(args.timeout, args.debug > 1)

        # Time control for summary purposes
        self._start_time = GameTime.get_time()
        self._end_time = None

        # Create the agent timer
        self._agent_watchdog = Watchdog(int(float(args.timeout)))
        signal.signal(signal.SIGINT, self._signal_handler)

        self.sumo_active = False
        self.scenario_index = 0
        self.synchronization = None 
        self.route_indexer = None 
        self.args = args 


    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Timeout: Agent took too long to setup")
        elif self.manager:
            self.manager.signal_handler(signum, frame)

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """

        self._cleanup()
        if hasattr(self, 'manager') and self.manager:
            del self.manager
        if hasattr(self, 'world') and self.world:
            del self.world

    def _cleanup(self):
        """
        Remove and destroy all actors
        """

        # Simulation still running and in synchronous mode?
        if self.manager and self.manager.get_running_status() \
                and hasattr(self, 'world') and self.world:
            # Reset to asynchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(False)

        if self.manager:
            self.manager.cleanup()

        CarlaDataProvider.cleanup()

        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                self.ego_vehicles[i].destroy()
                self.ego_vehicles[i] = None
        self.ego_vehicles = []

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        if hasattr(self, 'agent_instance') and self.agent_instance:
            self.agent_instance.destroy()
            self.agent_instance = None


    def _prepare_ego_vehicles(self, ego_vehicles, wait_for_ego_vehicles=False):
        """
        Spawn or update the ego vehicles
        """
        # print("Number of ego vehicles: ", len(ego_vehicles))
        if not wait_for_ego_vehicles:
            for vehicle in ego_vehicles:
                self.ego_vehicles.append(CarlaDataProvider.request_new_actor(vehicle.model,
                                                                             vehicle.transform,
                                                                             vehicle.rolename,
                                                                             color=vehicle.color,
                                                                             vehicle_category=vehicle.category))

        else:
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in ego_vehicles:
                    ego_vehicle_found = False
                    carla_vehicles = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')
                    for carla_vehicle in carla_vehicles:
                        if carla_vehicle.attributes['role_name'] == ego_vehicle.rolename:
                            ego_vehicle_found = True
                            self.ego_vehicles.append(carla_vehicle)
                            break
                    if not ego_vehicle_found:
                        ego_vehicle_missing = True
                        break

            for i, _ in enumerate(self.ego_vehicles):
                self.ego_vehicles[i].set_transform(ego_vehicles[i].transform)
                

        # for i, _ in enumerate(self.ego_vehicles): # TODO
            

        # sync state
        # CarlaDataProvider.get_world().tick()
        for i in range(5):
            self.synchronization.tick()

    def _load_and_wait_for_world(self, args, town, ego_vehicles=None):
        """
        Load a new CARLA world and provide data to CarlaDataProvider
        """

        self.world = self.client.load_world(town)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        self.world.reset_all_traffic_lights()

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(int(args.trafficManagerPort))

        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(int(args.trafficManagerSeed))

        # Wait for the world to be ready
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

        if CarlaDataProvider.get_map().name != town:
            raise Exception("The CARLA server uses the wrong map!"
                            "This scenario requires to use map {}".format(town))

        # Set up synchronization 
        # For spawning npcs (from spawn_npc_sumo.py)
        current_map = town
        # xodr_file = os.path.join(tmpdir, current_map.name + '.xodr')
        xodr_file = os.path.join(CARLA_HOME, "CarlaUE4/Content/Carla/Maps/OpenDrive", current_map +'.xodr')
        # current_map.save_to_disk(xodr_file)
        net_file = os.path.join(PROJECT_ROOT, "sumo_integration", "examples", "net", current_map + '.net.xml')
        new_net = False 

        # if not os.path.isfile(net_file):
        if town not in ["Town01", "Town04", "Town05"]:
            new_net = True
            netconvert_carla(xodr_file, net_file, guess_tls=True)
        basedir = os.path.join(PROJECT_ROOT, "sumo_integration")
        cfg_file = os.path.join(basedir,"examples", current_map + '.sumocfg')

        # if not os.path.isfile(cfg_file):
        if new_net: 
            vtypes_file = os.path.join(basedir, 'examples', 'carlavtypes.rou.xml')
            viewsettings_file = os.path.join(basedir, 'examples', 'viewsettings.xml')
            write_sumocfg_xml(cfg_file, net_file, vtypes_file, viewsettings_file, 0)

        if self.sumo_active: 
            self.sumo_simulation.close() 

        self.sumo_net = sumolib.net.readNet(net_file)
        self.sumo_simulation = SumoSimulation(cfg_file, 1/20, args.sumo_host,
                                     args.sumo_port, args.sumo_gui, 1)
        
        
        self.synchronization = SimulationSynchronization(self.sumo_simulation, self.carla_simulation, "carla",
                                                args.sync_vehicle_color, args.sync_vehicle_lights)
        
        for _ in range(5):
            self.synchronization.tick()


    def _load_and_prepare_scenario(self, args, config):
        """
        Load and run the scenario given by config.

        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.
        """
        crash_message = ""
        entry_status = "Started"

        print("\n\033[1m========= Preparing {} (repetition {}) =========".format(config.name, config.repetition_index))
        print("> Setting up the agent\033[0m")

        # Set up the user's agent, and the timer to avoid freezing the simulation
        try:
            self._agent_watchdog.start()
            # agent_class_name = getattr(self.module_agent, 'get_entry_point')()
            # self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config)
            self.agent_instance = AccelAgent(args.agent_config)
            self.agent_instance.synchronization = self.synchronization
            config.agent = self.agent_instance

            # Check and store the sensors
            if not self.sensors:
                self.sensors = self.agent_instance.sensors()
                track = self.agent_instance.track

                AgentWrapper.validate_sensor_configuration(self.sensors, track, args.track)

                self.sensor_icons = [sensors_to_icons[sensor['type']] for sensor in self.sensors]

            self._agent_watchdog.stop()

        except SensorConfigurationInvalid as e:
            # The sensors are invalid -> set the ejecution to rejected and stop
            print("\n\033[91mThe sensor's configuration used is invalid:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent's sensors were invalid"
            entry_status = "Rejected"

            self._register_statistics(config, args.checkpoint, entry_status, crash_message)
            self._cleanup()
            sys.exit(-1)

        except Exception as e:
            # The agent setup has failed -> start the next route
            print("\n\033[91mCould not set up the required agent:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent couldn't be set up"

            self._register_statistics(config, args.checkpoint, entry_status, crash_message)
            self._cleanup()
            return

        print("\033[1m> Loading the world\033[0m")

        # Load the world and the scenario
        try:
            if self.sumo_active:
                self.synchronization.sumo.close()
                self.sumo_active = False
            self._load_and_wait_for_world(args, config.town, config.ego_vehicles)
            self._prepare_ego_vehicles(config.ego_vehicles, False)
            self.agent_instance.synchronization = self.synchronization

            self.scenario = RouteScenario(world=self.world, config=config, debug_mode=args.debug)

            # Night mode
            if config.weather.sun_altitude_angle < 0.0:
                for vehicle in self.scenario.ego_vehicles:
                    vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))

            # Load scenario and run it
            if args.record:
                self.client.start_recorder("{}/{}_rep{}.log".format(args.record, config.name, config.repetition_index))
            self.manager.load_scenario(self.scenario, self.agent_instance, config.repetition_index)

            for i in range(5):
                self.synchronization.tick()

        except Exception as e:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"
            entry_status = "Crashed"

            # self._register_statistics(config, args.checkpoint, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            self._cleanup()
            sys.exit(-1)

        # self.manager.run_scenario()
        
        print("\033[1m> Scenario is loaded. \033[0m")
    

    def tick_scenario(self, action):

        results = None 

        # Run the scenario
        try:    
            results = self.manager.run_tick_cosim_rl(self.synchronization, action)
            return results

        except AgentError as e:
            # The agent has failed -> stop the route
            print("\n\033[91mStopping the route, the agent has crashed:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent crashed"

        except KeyboardInterrupt as e:
            print("\n\033[91mUser interrupted execution:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Cancelled by user"

        except Exception as e:
            print("\n\033[91mError during the simulation:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"
            entry_status = "Crashed"

        # if done: 
        # Stop the scenario
        try:
            print("\033[1m> Stopping the route\033[0m")
            self.manager.stop_scenario()

            if self.args.record:
                self.client.stop_recorder()

            # Remove all actors
            self.scenario.remove_all_actors()

            self._cleanup()
            self.synchronization.sumo.close()
            self.sumo_active = False

        except KeyboardInterrupt as e:
            print("\n\033[91mUser interrupted execution:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Cancelled by user"

        except Exception as e:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"

        print("Crashed; ", crash_message)
        if crash_message == "Cancelled by user":
            self._cleanup()
            sys.exit(-1)
        
        return None, None, None, True # rgb, state, player_ind

    def _initialize_route_indexer(self, args):
        self.route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)
        self.scenario_index = self.route_indexer._index 

        if args.resume:
            self.route_indexer.resume(args.checkpoint)
        else:
            self.route_indexer.save_state(args.checkpoint)

    def start_single_episode(self, restart=False): # called in reset()
        """ Run a single route scenario. 
        Automatically gets the next route from route indexer.
        """

        if self.route_indexer is None: 
            self._initialize_route_indexer(self.args)

        if restart: 
            self.route_indexer.set_route(self.scenario_index)

        if self.route_indexer.peek():
            # setup
            config = self.route_indexer.next()
            self.scenario_index = self.route_indexer._index
            self.config = config 

            # run
            self._load_and_prepare_scenario(self.args, config)

            self.route_indexer.save_state(self.args.checkpoint)
        else:
            self._initialize_route_indexer(self, self.args) # reinitialize route indexer
            if self.route_indexer.peek():
                # setup
                config = self.route_indexer.next()

                # run
                self._load_and_prepare_scenario(self.args, config)

                self.route_indexer.save_state(self.args.checkpoint)
        
        assert self.synchronization is not None, "Synchronization is not initialized"
        # assert self.sumo_active == True, "Sumo connection is not active"


def parse_args():
    description = "Co-sim acceleration optimization with CARLA. \n"
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    # parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')

    ############# Policy model training params ################
    # train params 
    # parser.add_argument('--max_epochs', type=int, default=1)
    # parser.add_argument('--save_dir', type=pathlib.Path, default='checkpoints')
    # parser.add_argument('--id', type=str, default=uuid.uuid4().hex)
    # parser.add_argument('--dagger_iterations', type=int, default=10)

    # parser.add_argument('--teacher_path', type=pathlib.Path, required=True)

    # Model args.
    # parser.add_argument('--heatmap_radius', type=int, default=5)
    # parser.add_argument('--sample_by', type=str, choices=['none', 'even', 'speed', 'steer'], default='steer')
    # parser.add_argument('--command_coefficient', type=float, default=0.1)
    # parser.add_argument('--reward_coefficient', type=float, default=0.1)
    # parser.add_argument('--temperature', type=float, default=5.0)
    # parser.add_argument('--hack', action='store_true', default=False)
    parser.add_argument('--reward_type', type=str, default="desired_velocity", choices=["avg_velocity", "desired_velocity"])

    # Data args.
    # parser.add_argument('--dataset_dir', type=pathlib.Path, required=True)
    # parser.add_argument('--batch_size', type=int, default=8)

    # Optimizer args.
    # parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--weight_decay', type=float, default=0.0)

    ############# Co-simulation Params ################
    # general parameters
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    # parser.add_argument('--port', default='2000', help='TCP port to listen to (default: 2000)')
    parser.add_argument('--trafficManagerPort', default='9000',
                        help='Port to use for the TrafficManager (default: 9000)')
    parser.add_argument('--trafficManagerSeed', default='0',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--debug', type=int, help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default="60.0",
                        help='Set the CARLA client timeout value in seconds')

    # simulation setup
    parser.add_argument('--routes',
                        type=str, 
                        default='./leaderboard/data/routes_training.xml',
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.')
    parser.add_argument('--scenarios',
                        type=str, 
                        default='leaderboard/data/all_towns_traffic_scenarios_public.json',
                        help='Name of the scenario annotation file to be mixed with the route.')
    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate", default="leaderboard/team_code/accel_agent.py") # required=True)
    parser.add_argument("--agent-config", type=str, help="Path to Agent's configuration file", default="")

    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")

    parser.add_argument('--carla-host',
                           metavar='H',
                           default='127.0.0.1',
                           help='IP of the carla host server (default: 127.0.0.1)')
    parser.add_argument('--carla-port',
                           metavar='P',
                           default=2000,
                           type=int,
                           help='TCP port to listen to (default: 2000)')
    parser.add_argument('--sumo-host',
                           metavar='H',
                           default=None,
                           help='IP of the sumo host server (default: 127.0.0.1)')
    parser.add_argument('--sumo-port',
                           metavar='P',
                           default=8813,
                           type=int,
                           help='TCP port to listen to (default: 8813)')
    parser.add_argument('--sumo-gui', action='store_true', help='run the gui version of sumo')
    parser.add_argument('--sync-vehicle-lights',
                           action='store_true',
                           help='synchronize vehicle lights state (default: False)')
    parser.add_argument('--sync-vehicle-color',
                           action='store_true',
                           help='synchronize vehicle color (default: False)')
    parser.add_argument('--sync-vehicle-all',
                           action='store_true',
                           help='synchronize all vehicle properties (default: False)')

    arguments = parser.parse_args()
    # arguments.teacher_path = arguments.teacher_path.resolve()
    # arguments.save_dir = arguments.save_dir.resolve() / arguments.id
    # arguments.save_dir.mkdir(parents=True, exist_ok=True)

    return arguments

if __name__ == "__main__":
    import gym
    from stable_baselines3 import PPO

    args = parse_args()

    # env = gym.make('CartPole-v1')
    env = CosimEnv(args)

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)

    model.save("accel_model_ppo")

    obs = env.reset()
    # for i in range(1000):
    #     action, _state = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    #     if done:
    #         obs = env.reset()
