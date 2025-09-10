import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from collections import Counter

from updated_mdp10 import (
    RingEnvironment, TransitionModel, RewardModel,
    ValueIteration, GlobalPolicy, State
)

def shannon_entropy_from_props(props):
    arr = np.asarray([p for p in props if p > 0.0], dtype=float)
    return float(-(arr * np.log(arr)).sum())

def run_simulation_with_tracking(env: RingEnvironment, global_policy: GlobalPolicy,
                                 transition_model: TransitionModel, steps: int = 30, 
                                 alpha: float = None, prestige_weight: float = None, 
                                 familiarity_weight: float = None):
    """Run simulation and track state evolution"""
    
    # display simulation parameters if provided
    if alpha is not None or prestige_weight is not None or familiarity_weight is not None:
        print(f"Running simulation with parameters:")
        if alpha is not None:
            print(f"  Alpha (neighborhood influence): {alpha:.3f}")
        if prestige_weight is not None:
            print(f"  Prestige weight: {prestige_weight:.3f}")
        if familiarity_weight is not None:
            print(f"  Familiarity weight: {familiarity_weight:.3f}")
    
    random.seed(3)
    
    # manually set the initial distribution
    initial_distribution = {
        State.SM: 0.60,
        State.SB: 0.30,
        State.DB: 0.05,
        State.DM: 0.05
    }
    
    # create global_state based on initial_distribution
    global_state = []
    for state, proportion in initial_distribution.items():
        count = int(round(proportion * env.num_agents))
        global_state.extend([state] * count)
    
    # handle any rounding differences
    while len(global_state) < env.num_agents:
        global_state.append(random.choice(list(initial_distribution.keys())))
    global_state = global_state[:env.num_agents]
    
    # shuffle to randomize positions
    random.shuffle(global_state)
    
    counter = Counter(global_state)
    state_distribution = {state.name: counter.get(state, 0) / env.num_agents for state in State}

    print("\n=== Simulation Start ===")
    print("Initial state distribution:")
    for state, proportion in state_distribution.items():
        print(f"{state}: {proportion:.3f}")
    history = []

    for step in range(steps):
        counter = Counter(global_state)
        state_counts = {s.name: counter.get(s, 0) / env.num_agents for s in State}
        state_counts["step"] = step
        history.append(state_counts)

        new_state = []
        for agent_id in range(env.num_agents):

            center_action = global_policy.get_action(agent_id, global_state, env)

            current_center_state = global_state[agent_id]

            trans_probs = transition_model.transition_rules.get(
                (current_center_state, center_action), {current_center_state: 1.0}
            )
            next_state = random.choices(
                list(trans_probs.keys()), 
                weights=list(trans_probs.values())
            )[0]
            new_state.append(next_state)

        global_state = new_state

    return pd.DataFrame(history)

def run_single_simulation(alpha: float, prestige_weight: float, 
                         familiarity_weight: float, steps: int, 
                         large_env_size: int):
    """Run a single simulation with specified parameters"""
    print(f"Running single simulation:")
    print(f"  Alpha: {alpha}")
    print(f"  Prestige weight: {prestige_weight}")
    print(f"  Familiarity weight: {familiarity_weight}")
    print(f"  Steps: {steps}")
    print(f"  Environment size: {large_env_size}")
    
    # create small environment for policy computation
    local_env = RingEnvironment(num_agents=3, neighborhood_radius=1)
    reward_model = RewardModel(local_env, alpha=alpha,
                               prestige_weight=prestige_weight,
                               familiarity_weight=familiarity_weight)
    transition_model = TransitionModel(local_env)
    vi = ValueIteration(local_env, transition_model, reward_model)
    values, policy = vi.iterate()

    # Create global policy and large environment
    global_policy = GlobalPolicy(local_env, policy)
    global_env = RingEnvironment(num_agents=large_env_size, neighborhood_radius=1)
    
    # run simulation with tracking
    df = run_simulation_with_tracking(global_env, global_policy, transition_model, 
                                      steps=steps, alpha=alpha, 
                                      prestige_weight=prestige_weight, 
                                      familiarity_weight=familiarity_weight)
    
    return df

def plot_state_evolution(df):
    plt.figure(figsize=(10, 6))
    for state in ['SM', 'SB', 'DB', 'DM']:
        plt.plot(df['step'], df[state], label=state)

    #plt.title('Agent State Distribution Over Time')
    plt.xlabel('Step')
    plt.ylabel('Proportion')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_parameter_grid_simulation():
    #alphas = [0, 0.5, 1.0]
    alphas = np.arange(0.0, 5.0, 5.0/30.0)
    #prestige_weights = np.arange(0.0, 2.0, 2.0/30.0)
    prestige_weights = [0.2]
    familiarity_weights = [1.0]
    #familiarity_weights = np.arange(0.0, 10, 10/30)

    param_grid = list(itertools.product(alphas, prestige_weights, familiarity_weights))
    results = []

    for alpha, prestige, familiarity in param_grid:
        print(f"Running simulation for alpha={alpha}, prestige={prestige}, familiarity={familiarity}")

        local_env = RingEnvironment(num_agents=3, neighborhood_radius=1)
        reward_model = RewardModel(local_env, alpha=alpha,
                                   prestige_weight=prestige,
                                   familiarity_weight=familiarity)
        transition_model = TransitionModel(local_env)
        vi = ValueIteration(local_env, transition_model, reward_model)
        values, policy = vi.iterate()

        global_policy = GlobalPolicy(local_env, policy)
        global_env = RingEnvironment(num_agents=100, neighborhood_radius=1)
        df = run_simulation_with_tracking(global_env, global_policy, transition_model, steps=30)

        final_row = df.iloc[-1]
        final_entropy = shannon_entropy_from_props([final_row[s] for s in ['SM','SB','DB','DM']])

        results.append({
            "alpha": alpha,
            "prestige_weight": prestige,
            "familiarity_weight": familiarity,
            **{state: final_row[state] for state in ['SM', 'SB', 'DB', 'DM']},
            "entropy": final_entropy
        })

    return pd.DataFrame(results)

def plot_results(df):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="alpha", y="DM", style="familiarity_weight", hue="prestige_weight", markers=True)
    #plt.title("Final Proportion of DM vs Alpha")
    plt.xlabel("Alpha")
    plt.ylabel("Proportion of DM at step 30")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_full_distribution(df):
    states = ["SM", "SB", "DB", "DM"]
    Y = df[states]
    X = df["alpha"]

    plt.figure(figsize=(10, 6))
    plt.stackplot(X, Y.T, labels=states, alpha=0.8)

    plt.xlabel("Alpha")
    plt.ylabel("Proportion")
    #plt.title("State Distribution over Alpha Sweep")
    plt.legend(title="Agent State", loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



""" def plot_entropy(df):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="alpha", y="entropy",
                 hue="prestige_weight", style="familiarity_weight", markers=True)
    plt.xlabel("Alpha (Neighborhood Influence)")
    plt.ylabel("Final Entropy (nats) at Step 30")
    plt.grid(True)
    plt.tight_layout()
    plt.show() """


if __name__ == "__main__":

    run_mode = "grid"  
    
    if run_mode == "single":
        df_evolution = run_single_simulation(
            alpha=0.5,              
            prestige_weight=0.2,    
            familiarity_weight=1.0, 
            steps=30,
            large_env_size=100
        )
        
       
        plot_state_evolution(df_evolution)
        
        print("\nFinal state distribution:")
        final_row = df_evolution.iloc[-1]
        for state in ['SM', 'SB', 'DB', 'DM']:
            print(f"{state}: {final_row[state]:.3f}")
        #final_entropy = shannon_entropy_from_props([final_row[s] for s in ['SM','SB','DB','DM']])
        #print(f"Final entropy: {final_entropy:.3f}")
            
        
    elif run_mode == "grid":
        df_results = run_parameter_grid_simulation()
        df_results.to_csv("simulation_results.csv", index=False)
        plot_results(df_results)

    else:
        df_results = run_parameter_grid_simulation()
        df_results.to_csv("simulation_results.csv", index=False)
        plot_full_distribution(df_results)

        