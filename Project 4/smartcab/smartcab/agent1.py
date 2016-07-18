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
        self.Q = {}
        # Values for Q - learning calculations
        self.learning_rate = 0.9
        self.discount_factor = 0.33
        self.default_Q = 1
        self.epsilon = 0.1
        # should yield an equal chance of a random action
        self.epsilon2 = 0.5

        # How successful is our agent
        self.success = 0
        self.total = 0
        self.trials = 1

        # How many penalties do we get?
        self.penalties = 0
        self.moves = 0
        self.net_reward = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        inputs['waypoint'] = self.next_waypoint
        self.state = tuple(sorted(inputs.items()))
        # TODO: Select action according to your policy
        Q, action = self.select_Q_action(self.state)

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

        if self.prev_state is not None:
            if (self.prev_state, self.prev_action) not in self.Q:
                self.Q[(self.prev_state, self.prev_action)] = self.default_Q

            # self.Q[(self.prev_state, self.prev_action)] = (1 - self.learning_rate) *\
            #     self.Q[(self.prev_state, self.prev_action)] + self.learning_rate *\
            #     (self.prev_reward + self.discount_factor *
            #         self.select_Q_action(self.state[0]))

                            # Correct method:
            self.Q[(self.prev_state,self.prev_action)] = (1 - self.learning_rate) * self.Q[(self.prev_state,self.prev_action)] + \
            self.learning_rate * (self.prev_reward + self.discount_factor * \
                self.select_Q_action(self.state)[0])

        self.prev_state = self.state
        self.prev_action = action
        self.prev_reward = reward

        self.env.status_text += ' ' + self._more_stats()

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
    
    def _more_stats(self):
        """Get additional stats"""
        return "success/total = {}/{} (net reward: {})\npenalties/moves (penalty rate): {}/{} ({})".format(
                self.success, self.total, self.net_reward, self.penalties, self.moves, round(float(self.penalties)/float(self.moves), 2))


    def select_Q_action(self, state):
        best_action = random.choice(Environment.valid_actions)
        if self.random_choice(self.epsilon):
            max_Q = self.get_Q(state, best_action)
        else:
            max_Q = -9999
            for action in Environment.valid_actions:
                Q = self.get_Q(state, action)
                if Q > max_Q:
                    max_Q = Q
                    best_action = action
                elif Q == max_Q:
                    if self.random_choice(self.epsilon2):
                        best_action = action
        return (max_Q, best_action)

    def get_Q(self, state, action):
        return self.Q.get((state, action), self.default_Q)

    def random_choice(self, epsilon=0.5):
        return random.random() < epsilon


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.3, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

if __name__ == '__main__':
    run()
