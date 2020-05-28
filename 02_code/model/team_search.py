import os
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN
from envs import Trainer

# Environment
env = DummyVecEnv([lambda: Trainer()])

# Train
model = PPO2(MlpPolicy, env, tensorboard_log='log')
model.learn(total_timesteps=100000)

# Save model
model.save('model')

# Play
episodes = 1
for e in range(episodes):
    obs = env.reset()
    done = False
    i = 0
    while not done:
        # Perform action
        action, _states = model.predict(obs)
        # Perform action
        obs, reward, done, _ = env.step(action)
        print('Iteration: {i} - Action: {a} - Reward: {r}'.format(i=i, a=action, r=reward))
        # Render environment
        env.render()
        # Iterate
        i += 1