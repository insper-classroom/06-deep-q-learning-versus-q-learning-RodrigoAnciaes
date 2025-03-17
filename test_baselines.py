from stable_baselines3 import DQN

model = DQN('MlpPolicy', 'CartPole-v1', verbose=1)
model.learn(total_timesteps=20000)
model.save('data/model_cart_pole')


# test if the model is working
obs = model.env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = model.env.step(action)
    model.env.render()
    if dones:
        break

model.env.close()
