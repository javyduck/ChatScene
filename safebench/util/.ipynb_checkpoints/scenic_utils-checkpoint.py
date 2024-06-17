
### Top-level functionality of the scenic package as a script:
### load a scenario and generate scenes in an infinite loop.
### modified from https://github.com/BerkeleyLearnVerify/Scenic/blob/main/src/scenic/__main__.py
### & https://github.com/BerkeleyLearnVerify/Scenic/blob/main/src/scenic/core/simulators.py

import os 
import random
import numpy as np 
import torch

import enum
import sys
import time
import argparse
import pygame
from collections import OrderedDict, defaultdict

if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata

import scenic.syntax.translator as translator
import scenic.core.errors as errors
from scenic.core.simulators import SimulationCreationError
from scenic.core.object_types import (enableDynamicProxyFor, setDynamicProxyFor,
                                      disableDynamicProxyFor)
from scenic.core.distributions import RejectionException
import scenic.core.dynamics as dynamics
from scenic.core.errors import RuntimeParseError, InvalidScenarioError, optionallyDebugRejection
from scenic.core.requirements import RequirementType
from scenic.core.vectors import Vector

def get_parser(scenicFile):
    parser = argparse.ArgumentParser(prog='scenic', add_help=False,
                                     usage='scenic [-h | --help] [options] FILE [options]',
                                     description='Sample from a Scenic scenario, optionally '
                                                 'running dynamic simulations.')

    mainOptions = parser.add_argument_group('main options')
    mainOptions.add_argument('-S', '--simulate', default=True,
                             help='run dynamic simulations from scenes '
                                  'instead of simply showing diagrams of scenes')
    mainOptions.add_argument('-s', '--seed', help='random seed', default=0, type=int)
    mainOptions.add_argument('-v', '--verbosity', help='verbosity level (default 1)',
                             type=int, choices=(0, 1, 2, 3), default=1)
    mainOptions.add_argument('-p', '--param', help='override a global parameter',
                             nargs=2, default=[], action='append', metavar=('PARAM', 'VALUE'))
    mainOptions.add_argument('-m', '--model', help='specify a Scenic world model', default='scenic.simulators.carla.model')
    mainOptions.add_argument('--scenario', default=None,
                             help='name of scenario to run (if file contains multiple)')

    # Simulation options
    simOpts = parser.add_argument_group('dynamic simulation options')
    simOpts.add_argument('--time', help='time bound for simulations (default none)',
                         type=int, default=10000)
    simOpts.add_argument('--count', help='number of successful simulations to run (default infinity)',
                         type=int, default=0)
    simOpts.add_argument('--max-sims-per-scene', type=int, default=1, metavar='N',
                         help='max # of rejected simulations before sampling a new scene (default 1)')

    # Interactive rendering options
    intOptions = parser.add_argument_group('static scene diagramming options')
    intOptions.add_argument('-d', '--delay', type=float,
                            help='loop automatically with this delay (in seconds) '
                                 'instead of waiting for the user to close the diagram')
    intOptions.add_argument('-z', '--zoom', type=float, default=1,
                            help='zoom expansion factor, or 0 to show the whole workspace (default 1)')

    # Debugging options
    debugOpts = parser.add_argument_group('debugging options')
    debugOpts.add_argument('--show-params', help='show values of global parameters',
                           action='store_true')
    debugOpts.add_argument('--show-records', help='show values of recorded expressions',
                           action='store_true')
    debugOpts.add_argument('-b', '--full-backtrace', help='show full internal backtraces',
                           action='store_true')
    debugOpts.add_argument('--pdb', action='store_true',
                           help='enter interactive debugger on errors (implies "-b")')
    debugOpts.add_argument('--pdb-on-reject', action='store_true',
                           help='enter interactive debugger on rejections (implies "-b")')
    ver = metadata.version('scenic')
    debugOpts.add_argument('--version', action='version', version=f'Scenic {ver}',
                           help='print Scenic version information and exit')
    debugOpts.add_argument('--dump-initial-python', help='dump initial translated Python',
                           action='store_true')
    debugOpts.add_argument('--dump-ast', help='dump final AST', action='store_true')
    debugOpts.add_argument('--dump-python', help='dump Python equivalent of final AST',
                           action='store_true')
    debugOpts.add_argument('--no-pruning', help='disable pruning', action='store_true')
    debugOpts.add_argument('--gather-stats', type=int, metavar='N',
                           help='collect timing statistics over this many scenes')
    
    parser.add_argument('--scenicFile', help='a Scenic file to run', default = scenicFile, metavar='FILE')

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help=argparse.SUPPRESS)

    # Parse arguments and set up configuration
    args = parser.parse_args(args=[])
    return args

class ScenicSimulator:
    def __init__(self, scenicFile, params):
        self.args = get_parser(scenicFile)
        delay = self.args.delay
        errors.showInternalBacktrace = self.args.full_backtrace
        if self.args.pdb:
            errors.postMortemDebugging = True
            errors.showInternalBacktrace = True
        if self.args.pdb_on_reject:
            errors.postMortemRejections = True
            errors.showInternalBacktrace = True
        translator.dumpTranslatedPython = self.args.dump_initial_python
        translator.dumpFinalAST = self.args.dump_ast
        translator.dumpASTPython = self.args.dump_python
        translator.verbosity = self.args.verbosity
        translator.usePruning = not self.args.no_pruning
#         if self.args.seed is not None and self.args.verbosity >= 1:
#             print(f'Using random seed = {self.args.seed}')
#             random.seed(self.args.seed)
        # Load scenario from file
        if self.args.verbosity >= 1:
            print('Beginning scenario construction...')
        startTime = time.time()
        self.scenario = errors.callBeginningScenicTrace(
            lambda: translator.scenarioFromFile(self.args.scenicFile,
                                                params=params,
                                                model=self.args.model,
                                                scenario=self.args.scenario)
        )
        self.opt_params, self.opt_record = self.get_params()
        
        totalTime = time.time() - startTime
        if self.args.verbosity >= 1:
            print(f'Scenario constructed in {totalTime:.2f} seconds.')
        self.simulator = errors.callBeginningScenicTrace(self.scenario.getSimulator)
        self.simulator.render = False
        
    def get_params(self):
        all_params = self.scenario.params
        opt_record = {}
        opt_params = {}
        for param in all_params.keys():
            if param.startswith('OPT'):
                opt_record[param] = []
                opt_params[param] = all_params[param]
                opt_params[param].min = opt_params[param].low
                opt_params[param].max = opt_params[param].high
        return opt_params, opt_record
    
    def record_params(self):
        all_params = self.scene.params
        for param in self.opt_record.keys():
            self.opt_record[param].append(all_params[param])
        print("Recording params...")
              
    def update_params(self):
        print("Updating params...")
        for param in self.opt_params.keys():
            if len(self.opt_record[param]) > 1:
                mean = np.mean(self.opt_record[param])
                std = np.std(self.opt_record[param])
                self.opt_params[param].low = round(max(mean - std, self.opt_params[param].min), 2)
                self.opt_params[param].high = round(min(mean + std, self.opt_params[param].max), 2)
#                 self.opt_record[param] = []
        self.scenario.params.update(self.opt_params)
        print(self.save_params())
        
    def load_params(self, params):
        print("Loading params...")
        for param in params.keys():
            self.opt_params[param].low = params[param]['low']
            self.opt_params[param].high = params[param]['high']
        self.scenario.params.update(self.opt_params)
        
    def save_params(self):
        print("Saving params...")
        save_params = {}
        for param in self.opt_params.keys():
            cur_param = self.opt_params[param]
            save_params[param] = {'low': cur_param.low, 'high': cur_param.high}
        return save_params
        
    def generateScene(self):
        scene, iterations = errors.callBeginningScenicTrace(
            lambda: self.scenario.generate(verbosity=self.args.verbosity)
        )
        return scene, iterations
    
    def setSimulation(self, scene):
        if self.args.verbosity >= 1:
            print(f'  Beginning simulation of {scene.dynamicScenario}...')
        try:     
            self.scene = scene
            self.simulation = self.simulator.createSimulation(scene, verbosity=self.args.verbosity)
        except SimulationCreationError as e:
            if self.args.verbosity >= 1:
                print(f'  Failed to create simulation: {e}')
            return False
        return True
        
    def runSimulation(self):
        """Run the simulation.
        Throws a RejectSimulationException if a requirement is violated.
        """
        maxSteps = self.args.time
        trajectory = self.simulation.trajectory
        if self.simulation.currentTime > 0:
            raise RuntimeError('tried to run a Simulation which has already run')
        assert len(trajectory) == 0
        actionSequence = []

        import scenic.syntax.veneer as veneer
        veneer.beginSimulation(self.simulation)
        dynamicScenario = self.simulation.scene.dynamicScenario

        # Initialize dynamic scenario
        dynamicScenario._start()

        # Give objects a chance to do any simulator-specific setup
        for obj in self.simulation.objects:
            if obj is self.simulation.objects[0]:
                continue
            obj.startDynamicSimulation()

        # Update all objects in case the simulator has adjusted any dynamic
        # properties during setup
        self.simulation.updateObjects()

        # Run simulation
        assert self.simulation.currentTime == 0
        terminationReason = None
        terminationType = None
        while True:
            yield self.simulation.currentTime
            if self.simulation.verbosity >= 3:
                print(f'    Time step {self.simulation.currentTime}:')

            # Run compose blocks of compositional scenarios
            # (and check if any requirements defined therein fail)
            terminationReason = dynamicScenario._step()
            terminationType = TerminationType.scenarioComplete

            # Record current state of the simulation
            self.simulation.recordCurrentState()

            # Run monitors
            newReason = dynamicScenario._runMonitors()
            if newReason is not None:
                terminationReason = newReason
                terminationType = TerminationType.terminatedByMonitor

            # "Always" and scenario-level requirements have been checked;
            # now safe to terminate if the top-level scenario has finished,
            # a monitor requested termination, or we've hit the timeout
            if terminationReason is not None:
                pass
            terminationReason = dynamicScenario._checkSimulationTerminationConditions()
            if terminationReason is not None:
                terminationType = TerminationType.simulationTerminationCondition
                pass
            if maxSteps and self.simulation.currentTime >= maxSteps:
                terminationReason = f'reached time limit ({maxSteps} steps)'
                terminationType = TerminationType.timeLimit
                pass

            # Compute the actions of the agents in this time step
            allActions = OrderedDict()
            schedule = self.simulation.scheduleForAgents()
            for agent in schedule:
                if agent is self.simulation.objects[0]:
                    continue

                behavior = agent.behavior
                if not behavior._runningIterator:   # TODO remove hack
                    behavior._start(agent)
                actions = behavior._step()
                if isinstance(actions, EndSimulationAction):
                    terminationReason = str(actions)
                    terminationType = TerminationType.terminatedByBehavior
                    break
                assert isinstance(actions, tuple)
                if len(actions) == 1 and isinstance(actions[0], (list, tuple)):
                    actions = tuple(actions[0])
                    
#                 if not self.simulation.actionsAreCompatible(agent, actions):
#                     raise InvalidScenarioError(f'agent {agent} tried incompatible '
#                                                f' action(s) {actions}')
                allActions[agent] = actions
            if terminationReason is not None:
                break

            # Execute the actions
            if self.simulation.verbosity >= 3:
                for agent, actions in allActions.items():
                    print(f'      Agent {agent} takes action(s) {actions}')
            actionSequence.append(allActions)
            self.simulation.executeActions(allActions)

            # Run the simulation for a single step and read its state back into Scenic
            # the step is controlled by safebench instead #
#             self.simulation.step()
            self.simulation.updateObjects()
            self.simulation.currentTime += 1
            
            # Package up simulation results into a compact object
            # update for every step #
            result = SimulationResult(trajectory, actionSequence, terminationType,
                                  terminationReason, self.simulation.records)
            self.simulation.result = result
        
    def endSimulation(self):
        # Stop all remaining scenarios
        # (and reject if some 'require eventually' condition was never satisfied)
        import scenic.syntax.veneer as veneer
        for scenario in tuple(veneer.runningScenarios):
            scenario._stop('simulation terminated')
            
        # If the simulation was terminated by an exception (including rejections),
        # some scenarios may still be running; we need to clean them up without
        # checking their requirements, which could raise rejection exceptions.
        
        for scenario in tuple(veneer.runningScenarios):
            scenario._stop('exception', quiet=True)
            
        if not hasattr(self, "simulation"):
            return 
        
        dynamicScenario = self.simulation.scene.dynamicScenario
        values = dynamicScenario._evaluateRecordedExprs(RequirementType.recordFinal)
        for name, val in values.items():
            self.simulation.records[name] = val
        
        ### destroy ###
        self.simulation.destroy()
        for obj in self.simulation.scene.objects:
            disableDynamicProxyFor(obj)
        for agent in self.simulation.agents:
            if agent.behavior._isRunning:
                agent.behavior._stop()
        for monitor in self.simulation.scene.monitors:
            if monitor._isRunning:
                monitor._stop()
        veneer.endSimulation(self.simulation)
        
    def destroy(self):
        self.simulator.destroy()
        
class Action:
    """An :term:`action` which can be taken by an agent for one step of a simulation."""
    def canBeTakenBy(self, agent):
        return True

    def applyTo(self, agent, simulation):
        raise NotImplementedError

class EndSimulationAction(Action):
    """Special action indicating it is time to end the simulation.
    Only for internal use.
    """
    def __init__(self, line):
        self.line = line

    def __str__(self):
        return f'"terminate" executed on line {self.line}'

class EndScenarioAction(Action):
    """Special action indicating it is time to end the current scenario.
    Only for internal use.
    """
    def __init__(self, line):
        self.line = line

    def __str__(self):
        return f'"terminate scenario" executed on line {self.line}'

@enum.unique
class TerminationType(enum.Enum):
    """Enum describing the possible ways a simulation can end."""
    #: Simulation reached the specified time limit.
    timeLimit = 'reached simulation time limit'
    #: The top-level scenario's :keyword:`compose` block finished executing.
    scenarioComplete = 'the top-level scenario finished'
    #: A user-specified termination condition was met.
    simulationTerminationCondition = 'a simulation termination condition was met'
    #: A :term:`monitor` used :keyword:`terminate` to end the simulation.
    terminatedByMonitor = 'a monitor terminated the simulation'
    #: A :term:`dynamic behavior` used :keyword:`terminate` to end the simulation.
    terminatedByBehavior = 'a behavior terminated the simulation'

class SimulationResult:
    """Result of running a simulation.
    Attributes:
        trajectory: A tuple giving for each time step the simulation's 'state': by
            default the positions of every object. See `Simulation.currentState`.
        finalState: The last 'state' of the simulation, as above.
        actions: A tuple giving for each time step a dict specifying for each agent the
            (possibly-empty) tuple of actions it took at that time step.
        terminationType (`TerminationType`): The way the simulation ended.
        terminationReason (str): A human-readable string giving the reason why the
            simulation ended, possibly including debugging info.
        records (dict): For each :keyword:`record` statement, the value or time series of
            values its expression took during the simulation.
    """
    def __init__(self, trajectory, actions, terminationType, terminationReason, records):
        self.trajectory = tuple(trajectory)
        assert self.trajectory
        self.finalState = self.trajectory[-1]
        self.actions = tuple(actions)
        self.terminationType = terminationType
        self.terminationReason = str(terminationReason)
        self.records = dict(records)
