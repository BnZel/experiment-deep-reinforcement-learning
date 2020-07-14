# Experimenting with Reinforcement Learning and Deep Q Learning
This was a way for me to understand the mechanics of RL and DQL, using OpenAI's custom gym environment.

This is still a **Work In Progress**.

## Intro
While learning the fundamentals of Q Learning, I referred to [DeepLizard's Tutorial](https://deeplizard.com/learn/video/nyjbcRQ-uQ8)

This environment was inspired by [Sentdex's Reinforcement Learning Tutorial](https://www.youtube.com/watch?v=yMk_XtIEzH8&list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7) with a minor modifications to fit OpenAI's environment

An overview of utilizing OpenAI's custom environment, I toke reference to [Cheesy AI's Tutorial](https://www.youtube.com/watch?v=ZxXKISVkH6Y)

## Environment
Following Sentdex's environment, it consists:
* A black background, initialized by the following class in **game.py**:   
```python  
# irrelevant code taken out to highlight the key point
# rest of the code will be avaliable in the files itself
class Game():
    def __init__(self):
        self.SIZE = 10
        self.env = np.zeros((self.SIZE,self.SIZE,3), dtype=np.uint8)
        
   def view(self):
        self.env[self.agent.y][self.agent.x] = self.COLOURS[self.AGENT_C]
        self.env[self.goal.y][self.goal.x] = self.COLOURS[self.GOAL_C]
        self.env[self.enemy.y][self.enemy.x] = self.COLOURS[self.ENEMY_C]

        # OpenCV's library to convert the zeroes to their respective colour
        # resizing the image for better view
        # and display
        img = Image.fromarray(self.env, "RGB")
        img.resize((300,300))

        cv2.imshow("image",np.array(img))
        cv2.waitKey(1)
```
* Zeroes represent the **RGB colour code of black** of the multidimensional array 
* 3 represents the **colour channels of RGB**
* **OpenCV's** library was used to represent the zeroes to their colour

The colour coding of the **Agent, Goal, and Enemy**:
``` python
        # initialized in the __init__ constructor
        self.AGENT_C = 1
        self.ENEMY_C = 2
        self.GOAL_C = 3
        self.COLOURS = {
        1: (255, 255, 255), # agent white
        2: (255, 0, 0),  # enemy red
        3: (0, 0, 255)  # goal blue
        }
        
        # in the view() function
        # assigning the x,y coordinates of the classes to predefined colours
        self.env[self.agent.y][self.agent.x] = self.COLOURS[self.AGENT_C]
        self.env[self.goal.y][self.goal.x] = self.COLOURS[self.GOAL_C]
        self.env[self.enemy.y][self.enemy.x] = self.COLOURS[self.ENEMY_C]
```
## States and Actions
* The Agent is allowed to have the following **actions**: (up,down,left,right)
* State or Observation space is represented by the distance between the: (agent - goal, agent - enemy)

The possible actions the agent can take:
```python 
# in game.py
# still work in progress
    def observe(self):       
        # based on distance of player - food, player - enemy 
        # observation space of (x1,y1)(x2,y2) 
        # distance between (player - goal)(player - enemy)
        observation = {}         
        for x1 in range(self.SIZE - 1):
            for y1 in range(self.SIZE - 1):
                 for x2 in range(self.SIZE - 1):                     
                    for y2 in range(self.SIZE - 1):                        
                        observation[((x1, y1),(x2, y2))] = self.obs                    
        return tuple(observation)
```
## Reward
Evaluating the agents action:
```python
      # set in game.py
      self.REWARD = 10
      self.ENEMY_PENALTY = 1000
      self.MOVE_PENALTY = 1
      
   def evaluate(self):
        # total amount of rewards, will be appended in the main.py
        self.current_reward = 0
        
        if self.agent.x == self.enemy.x and self.agent.y == self.enemy.y:
            self.current_reward = -self.ENEMY_PENALTY   
            
        elif self.agent.y == self.goal.x and self.agent.y == self.goal.y:
            self.current_reward = self.REWARD 
            
        else:
            self.current_reward = -self.MOVE_PENALTY        
        return self.current_reward
```
## Completing the episode
The game will finish once the x,y coordinates of the agent meets the enemy or goal 
```python
    def is_done(self):
        if (self.agent.x == self.enemy.x and self.agent.y == self.enemy.y) or (self.agent.x == self.goal.x and self.agent.y == self.goal.y):
            return True
        else:
            return False
```            
This will be passed through the step() function in custom_env.py        

## Custom Environment Setup
While referring to Cheesy AI's video for setup, to keep the main environment (custom_env.py) simplistic while implementing the game itself:
OpenAI's functions of (step, reset, render, and close)
```python
class CustomEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.game = Game()

    # all possible actions and environment data
    # 4: up, down, left, right
    self.action_space = spaces.Discrete(4)        
    
    self.observation_space = spaces.Box(np.zeros(9), np.zeros(9), dtype=np.uint8)

  def step(self, action):
    # Execute one time step within the environment
    self.game.action(action)
    reward = self.game.evaluate()
    obs = self.game.observe()
    done = self.game.is_done()
    return obs, reward, done, {}

  def reset(self):
    # Reset the state of the environment to an initial state
    del self.game
    self.game = Game()
    obs = self.game.observe()
    return obs

  def render(self, mode='human'):
    # Render the environment to the screen
    self.game.view()

  def close(self):
    return self.game.end()
```

## Utilizing Q-Learning
The common pattern I see coding the algorithm is to note the hyperparameters:
```python
    env = gym.make("foo-v0")

    num_episodes = 100 
    max_steps_per_episode = 50
    
    learning_rate = 0.1  
    
    # immediate and future rewards consideration
    # values between 0 - 1
    # where the agent will care more about the immediate reward over future rewards
    # in this case, the agent will take mostly of immediate rewards while considering future rewards
    discount_rate = 0.99
    
    # known as epsilon
    # starting state state to 1
    # meaning the agent will explore the environment first
    # later on the learning rate will decrease once the agent understands its surroundings
    # meaning it will exploit the values recorded in the table (q_table)
    exploration_rate = 1
    
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    
    # how much to decay 
    exploration_decay_rate = 0.001
 ```
 The Q-table storing the values of each state and action pairs, where the agent chooses based on the state that it's in:
 ```python
     # still work in progress 
     q_table = np.zeros([env.observation_space.shape[0], env.action_space.n])
```

The algorithm for Q-learning:
```python
    rewards_all_episodes = []
    for episode in range(num_episodes):
        print(f"starting game {episode}")
        state = env.reset()
        rewards_current_episode = 0
        done = False        
        
        # exploitation and exploration trade-off
        for step in range(max_steps_per_episode):
            exploration_rate_threshold = random.uniform(0,1)
            
            # exploit
            if exploration_rate_threshold > exploration_rate:
                action = np.argmax(q_table[state,:])
            else:
                # explore
                action = env.action_space.sample()

            # where game.py will start evaluating 
            new_state, reward, done, info = env.step(action)

            # using the bellman equation
            # matching to the right hand side
            q_table[state,action] = q_table[state,action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

            # monitoring new states
            state = new_state
            
            rewards_current_episode += reward

            if done == True:
                break

        exploration_rate = min_exploration_rate + (max_exploration_rate + min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
        rewards_all_episodes.append(rewards_current_episode)
 ```
 
 ## Conclusion
 My learning experience for Q-Learning has sparked some curiosity on some practical applications. 
 
 However Deep-Q-Learning would be the best way as the Q-table won't be as flooded and computationally heavy
 
 As there are still some things to fix and learn, I will continue to move forward to Deep-Q-Learning


