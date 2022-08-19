"""
Facilitates DAgger training routine with CARLA+SUMO co-simulation. 
Code is based on leaderboard evaluator by Codevilla et al.

"""
from __future__ import print_function

import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
from distutils.version import LooseVersion
import importlib
import os
import pkg_resources
import sys

import torch
import torchvision
import pytorch_lightning as pl

import pathlib
import wandb
import uuid

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import carla
import signal
import sumolib

from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

import lxml.etree as ET  # pylint: disable=wrong-import-position

from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper import  AgentWrapper, AgentError
from leaderboard.utils.statistics_manager import StatisticsManager
from leaderboard.utils.route_indexer import RouteIndexer

from carla_project.src.run_synchronization import SimulationSynchronization
from sumo_integration.bridge_helper import BridgeHelper  # pylint: disable=wrong-import-position
from sumo_integration.carla_simulation import CarlaSimulation  # pylint: disable=wrong-import-position
from sumo_integration.constants import INVALID_ACTOR_ID  # pylint: disable=wrong-import-position
from sumo_integration.sumo_simulation import SumoSimulation  # pylint: disable=wrong-import-position
from sumo_integration.util.netconvert_carla import netconvert_carla

from carla_project.src.traffic_img_model import TrafficImageModel

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


class DAggerRoutine(object):

    ego_vehicles = []

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    wait_for_world = 20.0  # in seconds
    frame_rate = 20.0      # in Hz

    def __init__(self, args, statistics_manager):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        self.statistics_manager = statistics_manager
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

        ## DAgger related attributes
        self.policy = None # Keep a reference to policy model 
        self.route_indexer = None



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

        if hasattr(self, 'statistics_manager') and self.statistics_manager:
            self.statistics_manager.scenario = None

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

        self.sumo_net = sumolib.net.readNet(net_file)
        self.sumo_simulation = SumoSimulation(cfg_file, 1/20, args.sumo_host,
                                     args.sumo_port, args.sumo_gui, 1)
        
        
        self.synchronization = SimulationSynchronization(self.sumo_simulation, self.carla_simulation, "carla",
                                                args.sync_vehicle_color, args.sync_vehicle_lights)
        
        self.synchronization.tick()

    def _register_statistics(self, config, checkpoint, entry_status, crash_message=""):
        """
        Computes and saved the simulation statistics
        """
        # register statistics
        current_stats_record = self.statistics_manager.compute_route_statistics(
            config,
            self.manager.scenario_duration_system,
            self.manager.scenario_duration_game,
            crash_message
        )

        print("\033[1m> Registering the route statistics\033[0m")
        self.statistics_manager.save_record(current_stats_record, config.index, checkpoint)
        self.statistics_manager.save_entry_status(entry_status, False, checkpoint)

    def _load_and_run_scenario(self, args, config, iteration:int, last_checkpoint:str, save=False):
        """
        Load and run the scenario given by config.

        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.
        """
        crash_message = ""
        entry_status = "Started"

        print("\n\033[1m========= Preparing {} (repetition {}) =========".format(config.name, config.repetition_index))
        print("> Setting up the agent\033[0m")

        # Prepare the statistics of the route
        self.statistics_manager.set_route(config.name, config.index)

        # Set up the user's agent, and the timer to avoid freezing the simulation
        try:
            self._agent_watchdog.start()
            agent_class_name = getattr(self.module_agent, 'get_entry_point')()

            print("Setting up agent: ", agent_class_name)
            self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config)

            if save:
                self.agent_instance._init_savedir(f"iteration{iteration:02d}")

            if last_checkpoint:
                self.agent_instance._init_policy(last_checkpoint)

            config.agent = self.agent_instance

            # Check and store the sensors
            if not self.sensors:
                self.sensors = self.agent_instance.sensors()
                track = self.agent_instance.track

                AgentWrapper.validate_sensor_configuration(self.sensors, track, args.track)

                self.sensor_icons = [sensors_to_icons[sensor['type']] for sensor in self.sensors]
                self.statistics_manager.save_sensors(self.sensor_icons, args.checkpoint)

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
            self._load_and_wait_for_world(args, config.town, config.ego_vehicles)
            self._prepare_ego_vehicles(config.ego_vehicles, False)
            self.agent_instance.synchronization = self.synchronization

            scenario = RouteScenario(world=self.world, config=config, debug_mode=args.debug)
            self.statistics_manager.set_scenario(scenario.scenario)

            # Night mode
            if config.weather.sun_altitude_angle < 0.0:
                for vehicle in scenario.ego_vehicles:
                    vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))

            # Load scenario and run it
            if args.record:
                self.client.start_recorder("{}/{}_rep{}.log".format(args.record, config.name, config.repetition_index))
            self.manager.load_scenario(scenario, self.agent_instance, config.repetition_index)

        except Exception as e:
            # The scenario is wrong -> set the execution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"
            entry_status = "Crashed"

            self._register_statistics(config, args.checkpoint, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            self._cleanup()
            sys.exit(-1)

        print("\033[1m> Running the route\033[0m")

        # Run the scenario
        try:
            
            # Do a few warm up ticks for synchronization
            for i in range(5):
                self.synchronization.tick()
                 
            self.manager.run_scenario_cosim(self.synchronization, max_ticks=8000)

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

        # Stop the scenario
        try:
            print("\033[1m> Stopping the route\033[0m")
            self.manager.stop_scenario()
            self._register_statistics(config, args.checkpoint, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            # Remove all actors
            scenario.remove_all_actors()

            self._cleanup()
            self.synchronization.close()

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

        if crash_message == "Cancelled by user":
            self._cleanup()
            sys.exit(-1)

    def _initialize_route_indexer(self, args):
        self.route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)
        if args.resume:
            self.route_indexer.resume(args.checkpoint)
            self.statistics_manager.resume(args.checkpoint)
        else:
            self.statistics_manager.clear_record(args.checkpoint)
            self.route_indexer.save_state(args.checkpoint)
        
    def _run_single_route(self, args, iteration, last_checkpoint, save=False):
        """ Run a single route scenario. 
        Automatically gets the next route from route indexer.
        """
        assert self.route_indexer is not None, "Route indexer is not initialized"

        if self.route_indexer.peek():
            # setup
            config = self.route_indexer.next()

            # run
            self._load_and_run_scenario(args, config, iteration, last_checkpoint, save=save)

            self.route_indexer.save_state(args.checkpoint)
        else:
            self._initialize_route_indexer(self, args) # reinitialize route indexer
            if self.route_indexer.peek():
                # setup
                config = self.route_indexer.next()

                # run
                self._load_and_run_scenario(args, config, iteration, last_checkpoint, save=save)

                self.route_indexer.save_state(args.checkpoint)

        # save global statistics
        print("\033[1m> Registering the global statistics\033[0m")
        global_stats_record = self.statistics_manager.compute_global_statistics(self.route_indexer.total)
        StatisticsManager.save_global_record(global_stats_record, self.sensor_icons, self.route_indexer.total, args.checkpoint)

    def _run_policy_cosim(self, args, save=False):
        """
        Run the challenge mode
        """

        assert self.route_indexer is not None, "Route indexer is not initialized"

        while self.route_indexer.peek():
            # setup
            config = self.route_indexer.next()

            # run
            self._load_and_run_scenario(args, config, save)

            self.route_indexer.save_state(args.checkpoint)

        # save global statistics
        print("\033[1m> Registering the global statistics\033[0m")
        global_stats_record = self.statistics_manager.compute_global_statistics(self.route_indexer.total)
        StatisticsManager.save_global_record(global_stats_record, self.sensor_icons, self.route_indexer.total, args.checkpoint)

    def _get_onpolicy_data(self, args, iteration, last_checkpoint): 
        for i in range(5):
            self._run_single_route(args, iteration, last_checkpoint, save=True)

    def train(self, args):
        
        self._initialize_route_indexer(args)

        # Get the latest checkpoint 
        try:
            last_checkpoint = sorted(args.save_dir.glob('*.ckpt'))[-1]
        except:
            last_checkpoint = None

        print("Latest checkpoint is: ", last_checkpoint)

        self.policy = TrafficImageModel(args, dagger=True)
        logger = WandbLogger(id=args.id, save_dir=str(args.save_dir), project='dagger_drive')
        checkpoint_callback = ModelCheckpoint(args.save_dir, 
                                                save_top_k=1)

        # Train on expert data first
        trainer = pl.Trainer(
                gpus=-1, max_epochs=args.max_epochs,
                resume_from_checkpoint=last_checkpoint,
                reload_dataloaders_every_n_epochs=args.max_epochs,
                logger=logger, checkpoint_callback=checkpoint_callback)

        trainer.fit(self.policy)

        for i in range(args.dagger_iterations):
            print("DAgger iteration ", i) 
            self.train_one_iteration(args, i, trainer, checkpoint_callback, logger)

        wandb.save(str(args.save_dir / '*.ckpt'))

    def train_one_iteration(self, args, i, trainer, checkpoint_callback, logger):
        try:
            last_checkpoint = sorted(args.save_dir.glob('*.ckpt'))[-1]
        except:
            last_checkpoint = None
        
        trainer.max_epochs += args.max_epochs 
        self._get_onpolicy_data(args, i, last_checkpoint) # collects on policy data
        trainer.fit(self.policy)

        # iteration_data_path = self._get_onpolicy_data(args, i) # collects on policy data

        # trainer = pl.Trainer(
        #         gpus=-1, max_epochs=args.max_epochs,
        #         resume_from_checkpoint=last_checkpoint,
        #         logger=logger, checkpoint_callback=checkpoint_callback)

        # trainer.fit(self.policy)

    def test(self, args):
        print("TODO")

def main():
    description = "DAgger data generation routine. Responsible for training DAgger on CARLA+SUMO co-simulation and iterative online data collection. \n"
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')

    ############# Policy model training params ################
    # train params 
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--save_dir', type=pathlib.Path, default='checkpoints')
    parser.add_argument('--id', type=str, default=uuid.uuid4().hex)
    parser.add_argument('--dagger_iterations', type=int, default=10)

    # parser.add_argument('--teacher_path', type=pathlib.Path, required=True)

    # Model args.
    parser.add_argument('--heatmap_radius', type=int, default=5)
    parser.add_argument('--sample_by', type=str, choices=['none', 'even', 'speed', 'steer'], default='even')
    parser.add_argument('--command_coefficient', type=float, default=0.1)
    parser.add_argument('--reward_coefficient', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=5.0)
    parser.add_argument('--hack', action='store_true', default=False)
    parser.add_argument('--reward_type', type=str, default="desired_velocity", choices=["avg_velocity", "desired_velocity"])

    # Data args.
    parser.add_argument('--dataset_dir', type=pathlib.Path, required=True)
    parser.add_argument('--batch_size', type=int, default=16)

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)

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
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.',
                        required=True)
    parser.add_argument('--scenarios',
                        help='Name of the scenario annotation file to be mixed with the route.',
                        required=True)
    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate", required=True)
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
    arguments.save_dir = arguments.save_dir.resolve() / arguments.id
    arguments.save_dir.mkdir(parents=True, exist_ok=True)

    statistics_manager = StatisticsManager()

    try:
        routine_manager = DAggerRoutine(arguments, statistics_manager)

        if arguments.mode == "train":
            routine_manager.train(arguments)
        else: 
            routine_manager.test(arguments)

        # routine_manager.run(arguments)

    except Exception as e:
        traceback.print_exc()
    finally:
        del routine_manager


if __name__ == '__main__':
    main()
