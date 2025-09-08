from dqn import DQN_Network
import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="rgb_array", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5)

# upper and lower bounds of env observation space
upper_bound = env.observation_space.high
lower_bound = env.observation_space.low

# Hyperparameters
GAMMA = 0.99
EPSILON = 1
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999
NUM_EPISODES = 6000
LEARNING_RATE = 0.001
HIDDEN_DIM = 128
UPDATE_TARGET_EVERY = 1000
BUFFER_SIZE = 100000
BATCH_SIZE = 128

# create agent
agent = DQN_Network(env, GAMMA, EPSILON, EPSILON_MIN, EPSILON_DECAY,
              NUM_EPISODES, LEARNING_RATE, lower_bound, upper_bound,
              HIDDEN_DIM, UPDATE_TARGET_EVERY, BUFFER_SIZE, BATCH_SIZE)

# train agent
agent.training()

# run learned policy
agent.save_policy_video(env, output_filename="learned_policy.mp4", episodes=8)