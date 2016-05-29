import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import defaultdict

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        self.success = 0
        self.total = 0
        self.break_rule = []
        self.break_plan = []
        self.break_rule_times = 0
        self.break_plan_times = 0

        # TODO: Initialize any additional variables here
        self.Q_table = {} # Initialize Q value
        self.alpha = 1 # learning_rate
        self.gamma = 0.5 # discount factor
        self.state_previous = None
        self.action_previous = None
        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state_previous
        self.action_previous
        break_rule_times = 0
        break_plan_times = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        #self.state = (inputs['light'], inputs['oncoming'], inputs['right'], inputs['left'], self.next_waypoint) # list format for sate, hashable for dict
        self.state = (inputs['light'], self.next_waypoint)
        
        # TODO: Select action according to your policy
        action = self.Best_Action(self.state, self.env.valid_actions)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        Q_value = self.Q_values(self.state, action)

        self.Q_table[(self.state, action)] = self.Q_values(self.state_previous, self.action_previous) + self.alpha * \
                                    (reward + self.gamma * Q_value - self.Q_values(self.state_previous, self.action_previous))

        self.state_previous = self.state
        self.action_previous = action

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        
        # Results statistics
        if reward == -0.5:
            self.break_plan_times += 1
        if reward == -1:
            self.break_rule_times += 1

        state_finish = False
        if reward >= 10:
            self.success += 1
            state_finish = True
        if deadline == 0 or reward >= 10:
            self.total += 1
            state_finish = True
            self.break_rule.append(self.break_rule_times)
            self.break_plan.append(self.break_plan_times)
        if state_finish:
            print 'success rate:', self.success, '/', self.total
            print 'break_plan:', self.break_plan
            print 'break_rule:', self.break_rule
            print 'learning rate:', self.alpha,'discount:', self.gamma


    def Q_values(self, state, action):
        """Get Q_value from Q_table according to (state,action) key"""
        if (state, action) not in self.Q_table:
            return 0
        return self.Q_table[(state,action)]

    def Best_Action(self, state, actions):
        """Return best_action with max Q_value according to the current state.
        state: current state
        actions: all avaliable actions."""
        best_qs = -999
        best_action = None
        for action in actions:
            qs = self.Q_values(state, action)
            if qs > best_qs:
                best_qs = qs
        for action in actions:
            if self.Q_values(state, action) == best_qs:
                best_action = action
        return best_action
               


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
