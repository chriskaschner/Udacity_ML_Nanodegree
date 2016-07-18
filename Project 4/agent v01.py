import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        trials = 1
        self.Q_init = 3
        # How many times agent able to reach the target for given trials?
        self.success = 0
        self.total = 0
        self.trials = trials

        # How many penalties does an agent get?
        self.penalties = 0
        self.moves = 0
        self.net_reward = 0
        
        # available actions
        self.actions = Environment.valid_actions
        
        # Initialize Q Table
        self.Q = {}
        for light in ['green', 'red']:
            for oncoming in ['oncoming', 'no_oncoming']:
                for waypoints in self.actions:
                    self.Q[light, oncoming, waypoints]= [self.Q_init] * len(self.actions)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state

        if inputs['oncoming'] is None:
            traffic = 'no_oncoming'
        else:
            traffic = 'oncoming'
        self.state = (inputs.values()[0], traffic, self.next_waypoint)

        # TODO: Select action according to your policy
        action = None

        max_Q = self.Q[self.state].index(max(self.Q[self.state]))
        #select action based on Q
        action = self.actions[max_Q]

        # Execute action and get reward
        reward = self.env.act(self, action)

                # Some stats
        self.net_reward += reward
        self.moves += 1
        if reward < 0:
            self.penalties+= 1

        add_total = False
        if deadline == 0:
            add_total = True
        if reward > 5:
            self.success += 1
            add_total = True
        if add_total:
            self.total += 1
            print self._more_stats()

        # TODO: Learn policy based on state, action, reward

        # Q-Learning process
        gamma = 0.45
        alpha = 0.2

        next_inputs = self.env.sense(self)
        if next_inputs['oncoming'] is None:
            next_traffic = 'no_oncoming'
        else:
            next_traffic = 'oncoming'
        next_next_waypoint = self.planner.next_waypoint()
        next_state = (next_inputs.values()[0], next_traffic, next_next_waypoint)
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

        # Update Q Table
        self.Q[self.state][self.actions.index(action)] = (1 - alpha) *\
            self.Q[self.state][self.actions.index(action)] +\
            (alpha * (reward + gamma * max(self.Q[next_state])))

    def _more_stats(self):
            """Get additional stats"""
            return "success/total = {}/{} of {} trials (net reward: {})\npenalties/moves (penalty rate): {}/{} ({})".format(
                    self.success, self.total, self.trials, self.net_reward, self.penalties, self.moves, round(float(self.penalties)/float(self.moves), 2))

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
