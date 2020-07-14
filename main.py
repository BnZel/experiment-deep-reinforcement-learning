import gym, gym_foo
import math, random, time
import numpy as np

def simulate():
    env = gym.make("foo-v0")
    for episode in range(100):
        state = env.reset()
        done = False
        print(f"***** EPISODE {episode+1} *****\n\n\n\n")

        for step in range(10):            
            env.render()
        
        action = np.argmax(q_table[state,:])
        new_state, reward, done, info = env.step(action)

        if done:
            if reward == 10:                
                print("****You reached the goal!****")
            else:
                print("****You hit an enemy!****")  
                time.sleep(5)          
            break

        state = new_state
    env.close()    

if __name__ == "__main__":
    env = gym.make("foo-v0")

    # env.reset()

    num_episodes = 100 #2500
    max_steps_per_episode = 50#500
    learning_rate = 0.1    
    discount_rate = 0.99
    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.001

    q_table = np.zeros([env.observation_space.shape[0], env.action_space.n])

    rewards_all_episodes = []
    for episode in range(num_episodes):
        print(f"starting game {episode}")
        state = env.reset()
        rewards_current_episode = 0
        done = False        
        
        for step in range(max_steps_per_episode):
            exploration_rate_threshold = random.uniform(0,1)
            if exploration_rate_threshold > exploration_rate:
                action = np.argmax(q_table[state,:])
            else:
                action = env.action_space.sample()

            new_state, reward, done, info = env.step(action)

            q_table[state,action] = q_table[state,action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

            state = new_state

            rewards_current_episode += reward

            if done == True:
                break

        exploration_rate = min_exploration_rate + (max_exploration_rate + min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
        rewards_all_episodes.append(rewards_current_episode)
    
    simulate()



