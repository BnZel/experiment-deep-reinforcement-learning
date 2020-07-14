import numpy as np
import cv2
from PIL import Image

class Square():
    def __init__(self):
        SIZE = 10        
        self.x = np.random.randint(0,SIZE)
        self.y = np.random.randint(0,SIZE)
        
    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

class Game():
    def __init__(self):
        self.current_reward = 0

        self.REWARD = 10
        self.ENEMY_PENALTY = 1000
        self.MOVE_PENALTY = 1

        self.SIZE = 10

        self.AGENT_C = 1
        self.ENEMY_C = 2
        self.GOAL_C = 3
        self.COLOURS = {
        1: (255, 255, 255), # agent white
        2: (255, 0, 0),  # enemy red
        3: (0, 0, 255)  # goal blue
        }
        
        self.agent = Square()
        self.goal = Square()
        self.enemy = Square()

        self.env = np.zeros((self.SIZE,self.SIZE,3), dtype=np.uint8)
        self.obs = (self.agent - self.goal, self.agent - self.enemy)

    def action(self, choice):
        # print(f"agent x,y : {self.agent.x, self.agent.y}")
        if choice == 0:
            self.agent.x += 1
            self.agent.y += 1
            
        elif choice == 1:
            self.agent.x += -1
            self.agent.y += -1

        elif choice == 2:
            self.agent.x += -1
            self.agent.y += 1

        elif choice == 3:
            self.agent.x += 1 
            self.agent.y += -1

        elif choice == 4:
            self.agent.x += 1
            self.agent.y += 0

        elif choice == 5:
            self.agent.x += -1
            self.agent.y += 0

        elif choice == 6:
            self.agent.x += 0
            self.agent.y += 1

        elif choice == 7:
            self.agent.x += 0
            self.agent.y += -1

        elif choice == 8:
            self.agent.x += 0
            self.agent.y += 0 
            
        # out of boundaries
        if self.agent.x < 0:
            self.agent.x = 0
        elif self.agent.x > self.SIZE - 1:
            self.agent.x = self.SIZE - 1

        if self.agent.y < 0:
            self.agent.y = 0
        elif self.agent.y > self.SIZE - 1:
            self.agent.y = self.SIZE - 1
    
    def view(self):
        self.env[self.agent.y][self.agent.x] = self.COLOURS[self.AGENT_C]
        self.env[self.goal.y][self.goal.x] = self.COLOURS[self.GOAL_C]
        self.env[self.enemy.y][self.enemy.x] = self.COLOURS[self.ENEMY_C]

        img = Image.fromarray(self.env, "RGB")
        img.resize((300,300))

        cv2.imshow("image",np.array(img))
        cv2.waitKey(1)

    def evaluate(self):
        self.current_reward = 0
        if self.agent.x == self.enemy.x and self.agent.y == self.enemy.y:
            self.current_reward = -self.ENEMY_PENALTY    
        elif self.agent.y == self.goal.x and self.agent.y == self.goal.y:
            self.current_reward = self.REWARD 
        else:
            self.current_reward = -self.MOVE_PENALTY
        return self.current_reward

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


    def is_done(self):
        if (self.agent.x == self.enemy.x and self.agent.y == self.enemy.y) or (self.agent.x == self.goal.x and self.agent.y == self.goal.y):
            return True
        else:
            return False

    def end(self):
        if self.current_reward == self.REWARD or self.current_reward == - self.ENEMY_PENALTY:
            if cv2.waitKey(500) & 0xFF == ord("q"):
                return None
        else:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return None

