import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import random
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(DQN,self).__init__()

        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN_Network():
    def __init__(self,env,gamma,epsilon,epsilon_min,epsilon_decay,
                 num_episodes,learning_rate,lower_bound,upper_bound,
                 hidden_dim,update_target_every,buffer_size,batch_size):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.num_episodes = num_episodes
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        # sum of all rewards in each episode
        self.episode_tot_reward = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:",self.device)

        #-------------------- DQN --------------------
        self.state_dim = env.observation_space.shape[0]
        self.hidden_dim = hidden_dim
        self.num_actions = self.env.action_space.n
        self.model = DQN(self.state_dim,self.hidden_dim,self.num_actions).to(self.device)

        # create target model and copy weights from model
        self.target_model = DQN(self.state_dim,self.hidden_dim,self.num_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        # set target model to evaluation only
        self.target_model.eval()

        self.learning_rate = learning_rate

        # counter for updating the weights of the target network
        self.update_target_every = update_target_every
        self.train_steps = 0

        # -------------------- Buffer --------------------
        self.buffer_size = buffer_size
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.batch_size = batch_size

        #---------------- Optimizer & loss function ----------------
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.MSELoss()

    def select_action(self, state):
        '''
        Selects an action using the epsilon-greedy strategy.

        Input:
            state (array-like): A list or array with 8 continuous values representing
                                the current observation from the LunarLander environment.

        Output:
            int: Selected action index (an integer in {0, 1, 2, 3}).
        '''

        # this random number is used in the epsilon-greedy approach
        random_number = np.random.rand()

        # if the random number is smaller than epsilon,
        # then select a random action,
        # otherwise select the action with the highest Q
        # value, as calculated by the DQN
        if random_number < self.epsilon:
            action = self.env.action_space.sample()
        else:
            # create a tensor that also has a batch dimension
            # as it is needed for pytorch (torch accepts input
            # [batch_size,input_size])
            state = torch.tensor(state,dtype=torch.float, device=self.device).unsqueeze(0)
            action = torch.argmax(self.model(state)).item()
        return action

    def train_step(self,batch):
        """
            Performs one training step using a mini-batch of transitions.

            Computes predicted Q-values from the model, target Q-values using the
            target network, and updates model weights via backpropagation.

            Args:
                batch (list of tuples): Each tuple is (state, action, reward, next_state, done),
                                        sampled from the replay buffer.

            Outputs:
                None. Updates the model parameters in-place.
            """
        
        # unpack the batch
        states, actions, rewards, next_states, terminal_state = zip(*batch)

        # convert to torch tensor
        # need to convert actions,rewards and terminal_state
        # to have shape [batch_size, 1] from [batch_size,]
        # for the loss calculation
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions,dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards,dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states),dtype=torch.float32, device=self.device)
        terminal_state = torch.tensor(terminal_state,dtype=torch.float32, device=self.device).unsqueeze(1)

        # forward pass
        q_values = self.model(states) # shape [batch_size,num_actions]
        q_values = q_values.gather(1, actions) # shape [batch_size,1]

        with torch.no_grad():
            next_q_values = self.target_model(next_states)  # shape [batch, num_actions]
            next_max_q = next_q_values.max(1, keepdim=True)[0] # shape [batch_size,1]

        # compute target
        target_q_values = rewards + self.gamma * next_max_q * (1 - terminal_state)

        # compute loss
        loss = self.loss_fn(q_values, target_q_values)

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def training(self):
        """
            Trains the DQN agent using experience replay.

            Fills the replay buffer with random actions (warm-up),
            then runs episodes using an epsilon-greedy policy.
            After each step, stores the transition and trains on
            a random batch from the buffer. Epsilon decays after
            each episode. Saves the model and reward plot at the end.

            Outputs:
                - Saves model weights to 'dqn_model.pth'
                - Saves training reward plot to 'training_rewards.png'
            """

        #-------------------- Buffer --------------------
        # warmup buffer with 1000 samples
        warm_up_samples = 1000
        state, _ = self.env.reset()
        for _ in range(warm_up_samples):
            # get random action
            action = self.env.action_space.sample()
            state_prime, reward, terminated, truncated, _ = self.env.step(action)
            terminal_state = terminated or truncated

            self.replay_buffer.append((state, action, reward, state_prime, terminal_state))
            state = state_prime

            if terminal_state:
                state, _ = self.env.reset()

        #-------------------- Training loop --------------------
        self.model.train()
        for episode in range(self.num_episodes):
            # reset the environment at the beginning of every episode
            state, _ = self.env.reset()
            terminal_state = False
            total_reward = 0

            while not terminal_state:
                # select action
                action = self.select_action(state)

                state_prime, reward, terminated, truncated, _ = self.env.step(action)
                terminal_state = terminated or truncated

                self.replay_buffer.append((state, action, reward, state_prime, terminal_state))
                if len(self.replay_buffer) >= self.batch_size:
                    batch = random.sample(self.replay_buffer, self.batch_size)
                    self.train_step(batch)

                total_reward += reward
                state = state_prime

                # update target network weights every 1000 steps
                self.train_steps += 1
                if self.train_steps % self.update_target_every == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

            # epsilon decay
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            self.episode_tot_reward.append(total_reward)

            if episode % 100 == 0:
                print(f"Episode {episode}: Total Reward = {total_reward}, Epsilon = {self.epsilon:.4f}")

            # reduce learning rate after episode 3000
            if episode == 3000:
                for g in self.optimizer.param_groups:
                    g['lr'] *= 0.1  # reduce by factor of 10

        # save model weights
        torch.save(self.model.state_dict(), "dqn_model.pth")

        # Average rewards per 100 episodes
        window_size = 100
        averaged_rewards = []
        for i in range(0, len(self.episode_tot_reward), window_size):
            averaged_rewards.append(np.mean(self.episode_tot_reward[i:i + window_size]))

        averaged_episodes = list(range(0, len(self.episode_tot_reward), window_size))

        plt.figure(figsize=(12, 6))
        plt.plot(averaged_episodes, averaged_rewards, label="Average Reward (per 100 episodes)", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("DQN Performance Over Episodes (smoothed)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("training_rewards.png")
        plt.show()

    def save_policy_video(self, env, output_filename="dqn_policy.mp4", episodes=8):
        """
        Runs the learned policy using the DQN model and saves the simulation as an MP4 video.

        Args:
            env: Gymnasium environment created with render_mode="rgb_array"
            output_filename (str): File path for saving the video
            episodes (int): Number of episodes to record
        """

        # load weights
        self.model.load_state_dict(torch.load("dqn_model.pth"))
        self.model.to(self.device)
        self.model.eval()

        frames = []
        labels = []
        rewards_per_episode = []

        # force greedy policy
        self.epsilon = 0.0

        for ep in range(episodes):
            done = False
            total_reward = 0

            state, _ = env.reset()
            frame = np.array(env.render())
            frames.append(frame)
            labels.append(f"Ep {ep}  Reward: {int(total_reward)}")

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                action = torch.argmax(self.model(state_tensor)).item()

                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward

                frame = np.array(env.render())
                frames.append(frame)
                labels.append(f"Ep {ep}  Reward: {int(total_reward)}")

            # add some sleep time for the last frame or each episode
            for _ in range(10):
                frames.append(frame)
                labels.append(f"Ep {ep}  Reward: {int(total_reward)}")

            rewards_per_episode.append(total_reward)
            print(f"Episode {ep}: Total Reward = {int(total_reward)}")

        env.close()

        # save frames as video
        fig, ax = plt.subplots()
        img = ax.imshow(frames[0])
        text = ax.text(10, 10, "", color="white", fontsize=12, weight="bold", backgroundcolor="black")
        plt.axis("off")

        def animate(i):
            img.set_array(frames[i])
            text.set_text(labels[i])
            return [img, text]

        ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50)
        ani.save(output_filename, writer="ffmpeg", fps=30)
        print(f"Saved policy video to '{output_filename}'")