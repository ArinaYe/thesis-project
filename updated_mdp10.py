import numpy as np
from typing import List, Tuple, Dict, Optional
from enum import Enum
import itertools
import pandas as pd
import random
from collections import Counter
import matplotlib.pyplot as plt

class State(Enum):
    """Individual agent states"""
    SM = 0  
    SB = 1  
    DB = 2  
    DM = 3  

class Action(Enum):
    """Individual agent actions"""
    A1 = 0
    A2 = 1

class RingEnvironment:
    """Environment representing the ring topology"""
    
    def __init__(self, num_agents: int, neighborhood_radius: int = 1):
        if num_agents < 3:
            raise ValueError("Ring must have at least 3 agents")
        if neighborhood_radius < 1 or 2 * neighborhood_radius + 1 > num_agents:
            raise ValueError(f"Invalid radius {neighborhood_radius} for {num_agents} agents")
            
        self.num_agents = num_agents
        self.neighborhood_radius = neighborhood_radius
        self.neighborhood_size = 2 * neighborhood_radius + 1
        
        self.states = list(State)
        self.actions = list(Action)
        self.state_space_size = len(self.states)
        self.action_space_size = len(self.actions)
        
        # Joint state/action spaces for neighborhoods
        self.joint_states = list(itertools.product(self.states, repeat=self.neighborhood_size))
        self.joint_actions = list(itertools.product(self.actions, repeat=self.neighborhood_size))
        self.joint_state_size = len(self.joint_states)
        self.joint_action_size = len(self.joint_actions)
        
    def get_neighbors(self, agent_id: int) -> List[int]:
        """Get all neighbors within radius of an agent in the ring"""
        neighbors = []
        for offset in range(-self.neighborhood_radius, self.neighborhood_radius + 1):
            neighbor_id = (agent_id + offset) % self.num_agents
            neighbors.append(neighbor_id)
        return neighbors
    
    def get_neighborhood_states(self, global_state: List[State], agent_id: int) -> Tuple[State, ...]:
        """Get joint state for neighborhood centered on agent_id"""
        neighbors = self.get_neighbors(agent_id)
        return tuple(global_state[neighbor] for neighbor in neighbors)
    
    def joint_state_to_index(self, joint_state: Tuple[State, ...]) -> int:
        """Convert joint state tuple to index"""
        try:
            return self.joint_states.index(joint_state)
        except ValueError:
            raise ValueError(f"Invalid joint state: {joint_state}")
    
    def joint_action_to_index(self, joint_action: Tuple[Action, ...]) -> int:
        """Convert joint action tuple to index"""
        try:
            return self.joint_actions.index(joint_action)
        except ValueError:
            raise ValueError(f"Invalid joint action: {joint_action}")

class TransitionModel:
    """Handles state transitions for agent neighborhoods"""
    
    def __init__(self, env: RingEnvironment):
        self.env = env
        self.transition_probs = np.zeros((
            env.joint_state_size,    # current joint state
            env.joint_action_size,   # joint action
            env.joint_state_size     # next joint state
        ))
        
        # Individual agent transition rules
        self.transition_rules = {
            (State.DM, Action.A1): {State.DM: 1.0},
            (State.SM, Action.A1): {State.SB: 0.5, State.SM: 0.5},
            (State.DB, Action.A1): {State.DM: 0.5, State.DB: 0.5},
            (State.SB, Action.A1): {State.DB: 0.5, State.SB: 0.5},
            (State.DM, Action.A2): {State.DB: 0.5, State.DM: 0.5},
            (State.SM, Action.A2): {State.SM: 1.0},
            (State.DB, Action.A2): {State.SB: 0.5, State.DB: 0.5},
            (State.SB, Action.A2): {State.SM: 0.5, State.SB: 0.5},
        }
        
        self._initialize_transitions()
    
    def _initialize_transitions(self):
        """Initialize transition probabilities based on individual agent rules"""
        for s_idx, joint_state in enumerate(self.env.joint_states):
            for a_idx, joint_action in enumerate(self.env.joint_actions):
                next_state_probs = self._compute_joint_transition_probs(joint_state, joint_action)
                
                for next_joint_state, prob in next_state_probs.items():
                    next_s_idx = self.env.joint_state_to_index(next_joint_state)
                    self.transition_probs[s_idx, a_idx, next_s_idx] = prob
    
    def _compute_joint_transition_probs(self, joint_state: Tuple[State, ...], 
                                      joint_action: Tuple[Action, ...]) -> Dict[Tuple[State, ...], float]:
        """Compute transition probabilities for joint state given joint action"""
        agent_transitions = []
        
        for i in range(self.env.neighborhood_size):
            current_state = joint_state[i]
            action = joint_action[i]
            
            if (current_state, action) in self.transition_rules:
                agent_transitions.append(self.transition_rules[(current_state, action)])
            else:
                agent_transitions.append({current_state: 1.0})
        
        # compute joint transition probabilities
        joint_transitions = {}
        
        # get all possible combinations of next states
        for next_states in itertools.product(*[list(trans.keys()) for trans in agent_transitions]):
            joint_prob = 1.0
            for i, next_state in enumerate(next_states):
                joint_prob *= agent_transitions[i][next_state]
            
            if next_states in joint_transitions:
                joint_transitions[next_states] += joint_prob
            else:
                joint_transitions[next_states] = joint_prob
        
        return joint_transitions
    
    def get_transition_prob(self, current_state: int, action: int, next_state: int) -> float:
        """Get transition probability P(s'|s,a)"""
        return self.transition_probs[current_state, action, next_state]


class RewardModel:
    """Handles reward computation for agent neighborhoods"""
    
    def __init__(self, env: RingEnvironment, alpha: float = 0.8, 
                 prestige_weight: float = 0.2, familiarity_weight: float = 1.0):
        self.env = env
        self.alpha = alpha  # weight of the neighborhood effect
        self.center_idx = env.neighborhood_radius  
        self.prestige_weight = prestige_weight  
        self.familiarity_weight = familiarity_weight  
        
        self.preferred_actions = {
            State.DM: Action.A1,
            State.DB: Action.A1,
            State.SM: Action.A2,
            State.SB: Action.A2
        }
        # initialize reward matrix
        self.rewards = np.zeros((env.joint_state_size, env.joint_action_size))
        self.initialize_rewards()

    def initialize_rewards(self):
        """Initialize reward function for all state-action pairs"""
        for s_idx, s in enumerate(self.env.joint_states):
            for a_idx, a in enumerate(self.env.joint_actions):
                self.rewards[s_idx, a_idx] = self.compute_reward(s, a)

    def compute_reward(self, joint_state: Tuple[State, ...], 
                      joint_action: Tuple[Action, ...]) -> float:
        center_action = joint_action[self.center_idx]
        center_state = joint_state[self.center_idx]
        
        individual_reward = self.compute_individual_reward(center_state, center_action)
        
        alignment_score = self.compute_alignment(joint_state, center_action)
        
        return individual_reward + self.alpha * alignment_score

    def compute_individual_reward(self, state: State, action: Action) -> float:
        """Compute individual reward (prestige + familiarity)"""
        prestige = self.compute_prestige_reward(state, action)
        familiarity = self.compute_familiarity_reward(state, action)
        return prestige + familiarity

    def compute_prestige_reward(self, state: State, action: Action) -> float:
        """Compute prestige-based reward"""
        if action == Action.A1:
            return self.prestige_weight
        elif action == Action.A2:
            return -self.prestige_weight
        return 0.0

    def compute_familiarity_reward(self, state: State, action: Action) -> float:
        """Compute familiarity-based reward"""
        
        preferred_action = self.preferred_actions.get(state)
        if preferred_action is None:
            return 0.0
            
        if action == preferred_action:
            return self.familiarity_weight
        else:
            return -self.familiarity_weight

    def compute_alignment(self, joint_state: Tuple[State, ...], center_action: Action) -> float:
        alignment = 0
        neighbor_count = 0

        for i, neighbor_state in enumerate(joint_state):
            if i == self.center_idx:  
                continue
                
            
            if center_action == Action.A1 and neighbor_state in [State.DM, State.DB]:
                alignment += 1
            elif center_action == Action.A2 and neighbor_state in [State.SM, State.SB]:
                alignment += 1
                
            neighbor_count += 1

        return (alignment / neighbor_count) if neighbor_count > 0 else 0.0

    def get_reward(self, state_idx: int, action_idx: int) -> float:
        if not (0 <= state_idx < self.rewards.shape[0]):
            raise ValueError(f"State index {state_idx} out of bounds")
        if not (0 <= action_idx < self.rewards.shape[1]):
            raise ValueError(f"Action index {action_idx} out of bounds")
            
        return self.rewards[state_idx, action_idx]


class ValueIteration:
    """Value iteration algorithm for the ring MDP"""
    
    def __init__(self, env: RingEnvironment, transition_model: TransitionModel, 
                 reward_model: RewardModel, discount_factor: float = 0.9, 
                 tolerance: float = 1e-6):
        self.env = env
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.gamma = discount_factor
        self.tolerance = tolerance
        
        # value function and policy
        self.values = np.zeros(env.joint_state_size)
        self.policy = np.zeros(env.joint_state_size, dtype=int)
        
    def bellman_update(self, state: int) -> Tuple[float, int]:
        """Perform Bellman update for a single state"""
        max_value = float('-inf')
        best_action = 0
        
        for action in range(self.env.joint_action_size):
            expected_value = self.reward_model.get_reward(state, action)
            for next_state in range(self.env.joint_state_size):
                prob = self.transition_model.get_transition_prob(state, action, next_state)
                expected_value += self.gamma * prob * self.values[next_state]
            
            if expected_value > max_value:
                max_value = expected_value
                best_action = action
        
        return max_value, best_action

    def iterate(self, max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Run value iteration until convergence"""
        for iteration in range(max_iterations):
            new_values = np.zeros_like(self.values)
            new_policy = np.zeros_like(self.policy)
            delta = 0

            for state in range(self.env.joint_state_size):
                v, a = self.bellman_update(state)
                new_values[state] = v
                new_policy[state] = a
                delta = max(delta, abs(v - self.values[state]))

            self.values = new_values
            self.policy = new_policy

            if delta < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
        else:
            print(f"Did not converge after {max_iterations} iterations")
        
        return self.values, self.policy

class GlobalPolicy:
    """Manages policy for all agents in the ring"""
    
    def __init__(self, local_env: RingEnvironment, policy: np.ndarray):
        """
        Args:
            local_env: The small environment used for policy computation
            policy: Array where policy[state_idx] = action_idx for joint states
        """
        self.local_env = local_env  # environment used for policy computation
        self.policy = policy  # single policy mapping joint states to joint actions
    
    def get_action(self, agent_id: int, global_state: List[State], 
                   global_env: RingEnvironment) -> Action:
        """
        Get action for a specific agent given global state
        
        Args:
            agent_id: ID of the agent we want an action for
            global_state: Current state of all agents in the large ring
            global_env: The large ring environment
        """
        neighborhood_state = global_env.get_neighborhood_states(global_state, agent_id)
        
        state_idx = self.local_env.joint_state_to_index(neighborhood_state)
        
        action_idx = self.policy[state_idx]
        joint_action = self.local_env.joint_actions[action_idx]
        
        center_idx = self.local_env.neighborhood_radius
        return joint_action[center_idx]


def main():
    """Main function to run the MDP simulation"""
    from updated_simulation import run_single_simulation, plot_state_evolution
    
    
    df_evolution = run_single_simulation(
        alpha=1.0,
        prestige_weight=0.2, 
        familiarity_weight=1.0,
        steps=30,
        large_env_size=100
    )
    

    plot_state_evolution(df_evolution)
    
    print("Final state distribution:")
    final_row = df_evolution.iloc[-1]
    for state in ['SM', 'SB', 'DB', 'DM']:
        print(f"{state}: {final_row[state]:.3f}")

if __name__ == "__main__":
    main()