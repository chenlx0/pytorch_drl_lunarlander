import gym
import dqn
import pg
import random
import torch
import numpy as np

def same_seed(seed): 
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_agent_by_name(name: str):
    if name == 'dqn':
        return dqn.DQNAgent
    elif name == 'pg':
        return pg.ReinforceAgent
    return None

env = gym.make("LunarLander-v2", render_mode="human")
env.action_space.seed(114514)

state, info = env.reset(seed=114514)
action_dim = env.action_space.n
state_dim = len(state)

if __name__ == "__main__":
    print("Begin! state dimension:%d, action dimension:%d" % (state_dim, action_dim))
    same_seed(19260817)
    epsidoes = 50000
    steps = 0
    agent = get_agent_by_name('dqn')(state_dim, action_dim)
    agent.enable_train()
    for i in range(epsidoes):
        print("round: %d" % (i))
        total_reward = 0.0
        terminated, truncated = False, False
        last_action, action, last_state = None, 0, None
        while not terminated and not truncated:
            state, reward, terminated, truncated, info = env.step(action)
            steps += 1
            total_reward += reward
            if last_state is not None:
                agent.add_to_memory(last_state, action, reward, state, terminated or truncated) 
            # training with interacting environments
            action = agent.select_action(state)
            last_state = state

            if steps % 10 == 0:
                agent.learn_by_replay()

        state, info = env.reset()
        print("round %d, steps: %d, rewards: %f" % (i, steps, total_reward))
        agent.copy_new_params()

    env.close()
