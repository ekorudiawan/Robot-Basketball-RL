import pygame 
import numpy as np 
import random 
import time 
import os
import matplotlib.pyplot as plt 
import tensorflow as tf

class Logger(object):
    def __init__(self, log_dir1, log_dir2):
        self.writer1 = tf.summary.create_file_writer(log_dir1)
        self.writer2 = tf.summary.create_file_writer(log_dir2)

    def log_scalar(self, tag, value1, value2, step):
        with self.writer1.as_default():
            tf.summary.scalar(tag, value1, step=step)
        with self.writer2.as_default():
            tf.summary.scalar(tag, value2, step=step)

class BasketballEnv:
    def __init__(self, width=9, height=6, n_opponents=5, render=False, version=0):
        self.width = width
        self.height = height
        self.scale = 100 
        self.n_opponents = n_opponents
        # Color
        self.ball_color = (168, 92, 50)
        self.white_color = (255, 255, 255)
        self.field_color = (50, 168, 52)
        self.basket_color = (168, 58, 50)
        self.robot_color = (27, 34, 48)
        self.opponent_color = (84, 3, 3)
        self.after_shoot = 1
        self.ball_to_basket = 0
        self.robot_pos_static = True
        self.version = version
        self.total_shoot_success = 0
        self.total_shoot_fail = 0
        # PyGame
        self.render_ui = render 
        if self.render_ui:
            pygame.init()
            pygame.display.set_caption("Basketball-V0")
            self.window_size = [self.width*self.scale, self.height*self.scale]
            self.screen = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()
            pygame.font.init() 
            self.font = pygame.font.SysFont('Comic Sans MS', 20)
            self.robot_text = self.font.render('ROBOT', False, (192, 197, 207))
            self.ball_text = self.font.render('BALL', False, (192, 197, 207))
            self.opp_text = self.font.render('OPP', False, (192, 197, 207))
        # Default Position
        self.basket_pos = [self.width-1, (self.height-1)//2]
        self.ball_pos = [0,(self.height-1)//2]

        if self.robot_pos_static:
            self.robot_pos = [0, 0]
        else:
            self.robot_pos = [random.randint(0, self.width-1), random.randint(0, self.height-1)]

        self.opponent_pos = []
        random.seed(42)
        for i in range(self.n_opponents):
            random.seed(time.clock())
            done = False
            while not done:
                pos = [random.randint(0, self.width-1), random.randint(0, self.height-1)]
                if pos not in self.opponent_pos:
                    if pos != self.robot_pos and pos != self.ball_pos:
                        self.opponent_pos.append(pos)
                        done = True 

        self.action_dict = {0: "UP", 
                            1: "LEFT",
                            2: "DOWN",
                            3: "RIGHT",
                            4: "DRIBBLE-UP",
                            5: "DRIBBLE-LEFT",
                            6: "DRIBBLE-DOWN",
                            7: "DRIBBLE-RIGHT",
                            8: "SHOOT"}

    def target_empty(self):
        result = True
        for i in range(self.n_opponents):
            if self.opponent_pos[i][0] == self.robot_pos[0] and self.opponent_pos[i][1] == self.robot_pos[1]:
                result = False 
        return result

    def ball_out(self):
        if self.ball_pos[0] < 0 or self.ball_pos[0] >= self.width or self.ball_pos[1] < 0 or self.ball_pos[1] >= self.height:
            return True
        else:
            return False

    def leave_field(self):
        if self.robot_pos[0] < 0 or self.robot_pos[0] >= self.width or self.robot_pos[1] < 0 or self.robot_pos[1] >= self.height:
            return True
        else:
            return False
    
    def drible_possible(self):
        if self.robot_pos[0] == self.ball_pos[0] and self.robot_pos[1] == self.ball_pos[1]:
            return True
        else:
            return False

    def distance_to_basket(self):
        distance = np.sqrt((self.basket_pos[0]-self.robot_pos[0])**2 + (self.basket_pos[1]-self.robot_pos[1])**2)
        return distance

    def distance_ball_to_basket(self):
        distance = np.sqrt((self.basket_pos[0]-self.ball_pos[0])**2 + (self.ball_pos[1]-self.ball_pos[1])**2)
        return distance

    def step(self, action):
        done = False
        reward = 0

        robot_prev_pos = self.robot_pos.copy()
        ball_prev_pos = self.ball_pos.copy()

        move = 0
        if self.version == 0:
            move = 1
        else:
            move_choice = [1,2]
            move = np.random.choice(move_choice, p=[0.6, 0.4])

        if self.action_dict[action] == "UP":
            self.robot_pos[1] -= move
        elif self.action_dict[action] == "LEFT":
            self.robot_pos[0] -= move
        elif self.action_dict[action] == "DOWN":
            self.robot_pos[1] += move
        elif self.action_dict[action] == "RIGHT":
            self.robot_pos[0] += move
        elif self.action_dict[action] == "DRIBBLE-UP":
            if self.drible_possible():
                self.robot_pos[1] -= move
                self.ball_pos[1] -= move
        elif self.action_dict[action] == "DRIBBLE-LEFT":
            if self.drible_possible():
                self.robot_pos[0] -= move
                self.ball_pos[0] -= move
        elif self.action_dict[action] == "DRIBBLE-DOWN":
            if self.drible_possible():
                self.robot_pos[1] += move
                self.ball_pos[1] += move
        elif self.action_dict[action] == "DRIBBLE-RIGHT":
            if self.drible_possible():
                self.robot_pos[0] += move
                self.ball_pos[0] += move
        elif self.action_dict[action] == "SHOOT":
            if self.drible_possible():
                # check distance
                dist = self.distance_to_basket() 
                shoot_choice = ["success","fail"]
                shoot_prob = 0
                shoot_reward_success = 0
                shoot_reward_fail = 0
                if dist < 1:
                    shoot_prob = 0.9
                    shoot_reward_success = 100
                    shoot_reward_fail = 50
                elif dist >= 1 and dist < 3:
                    shoot_prob = 0.66
                    shoot_reward_success = 200
                    shoot_reward_fail = 30
                elif dist >= 3 and dist < 4:
                    shoot_prob = 0.1
                    shoot_reward_success = 300
                    shoot_reward_fail = 20
                else:
                    shoot_prob = 0.0
                    shoot_reward_fail = -100

                shoot_result = np.random.choice(shoot_choice, p=[shoot_prob, 1-shoot_prob])
                if shoot_result == "success":
                    reward += shoot_reward_success 
                    self.ball_pos = self.basket_pos.copy()
                    self.total_shoot_success += 1
                    done = True

                else:
                    reward += shoot_reward_fail
                    self.ball_pos[0] = 0
                    self.ball_pos[1] = (self.height-1)//2
                    self.after_shoot = 2
                    self.total_shoot_fail += 1
                print("Shoot Distance :", "{:.2f}".format(dist)," Result :", shoot_result)

        if self.leave_field():
            done = True
            reward -= 100
        
        # If target not empy, back to previous position
        if not self.target_empty():
            self.robot_pos = robot_prev_pos.copy()
            self.ball_pos = ball_prev_pos.copy()

        # If ball out 
        if self.ball_out():
            self.ball_pos[0] = 0
            self.ball_pos[1] = (self.height-1)//2
        
        holding_ball = 1 
        if self.drible_possible():
            holding_ball = 2

        dist = self.distance_ball_to_basket()
        if dist < self.ball_to_basket:
            if dist != 0:
                reward += 100/dist
            self.ball_to_basket = dist

        observation = np.array([self.robot_pos[0], self.robot_pos[1], holding_ball, self.after_shoot])

        return observation, done, reward

    def reset(self):
        self.total_shoot_success = 0
        self.total_shoot_fail = 0
        if self.robot_pos_static:
            self.robot_pos = [0, 0]
        else:
            self.robot_pos = [random.randint(0, self.width-1), random.randint(0, self.height-1)]
        
        if self.version == 2:
            self.opponent_pos = []
            random.seed(42)
            for i in range(self.n_opponents):
                random.seed(time.clock())
                done = False
                while not done:
                    pos = [random.randint(0, self.width-1), random.randint(0, self.height-1)]
                    if pos not in self.opponent_pos:
                        if pos != self.robot_pos and pos != self.ball_pos:
                            self.opponent_pos.append(pos)
                            done = True 

        self.after_shoot = 1
        self.ball_pos = [0,(self.height-1)//2]
        holding_ball = 1
        self.ball_to_basket = self.distance_ball_to_basket()
        observation = np.array([self.robot_pos[0], self.robot_pos[1], holding_ball, self.after_shoot])
        return observation

    def draw_grid_cell(self):
        for i in range(0, self.window_size[0], self.scale):
            for j in range(0, self.window_size[1], self.scale):
                pygame.draw.rect(self.screen, self.white_color, [i, j, self.scale, self.scale], 3)

    def draw_basket(self):
        pygame.draw.rect(self.screen, self.basket_color, [self.basket_pos[0]*self.scale, self.basket_pos[1]*self.scale, self.scale, self.scale])

    def draw_ball(self):
        x = (self.ball_pos[0]*self.scale) + int((3/4)*self.scale)
        y = (self.ball_pos[1]*self.scale) + int((1/4)*self.scale)
        r = self.scale//4
        pygame.draw.circle(self.screen, self.ball_color, [x, y], r)
        self.screen.blit(self.ball_text,(x-int((1/6)*self.scale), y-int((1/8)*self.scale)))

    def draw_robot(self):
        x = self.robot_pos[0]*self.scale
        y = self.robot_pos[1]*self.scale
        w = self.scale//2
        h = self.scale//2
        pygame.draw.rect(self.screen, self.robot_color, [x, y, w, h])
        self.screen.blit(self.robot_text,(x+int((1/32)*self.scale), y+int((2/16)*self.scale)))

    def draw_opponents(self):
        for i in range(len(self.opponent_pos)):
            x = (self.opponent_pos[i][0]*self.scale) + self.scale//2
            y = (self.opponent_pos[i][1]*self.scale) + self.scale//2
            w = self.scale//2
            h = self.scale//2
            pygame.draw.rect(self.screen, self.opponent_color, [x, y, w, h])
            self.screen.blit(self.opp_text,(x+int(1/8*self.scale), y+int(1/8*self.scale)))

    def render(self):
        self.screen.fill(self.field_color)
        self.draw_grid_cell()
        self.draw_basket()
        self.draw_ball()
        self.draw_robot()
        self.draw_opponents()
        self.clock.tick(60)
        pygame.display.flip()

class QLearning:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.95, epsilon_decay=0.99, epsilon_min=0.01, render=False, logging=False):
        self.alpha = alpha 
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.epsilon_decay = epsilon_decay 
        self.epsilon_min = epsilon_min
        self.render = render
        self.env = BasketballEnv(width=9, height=6, n_opponents=5, render=render, version=0)
        self.n_states = self.env.width * self.env.height * 2 * 2
        self.n_actions = len(self.env.action_dict)
        self.Q_table = np.zeros((self.n_states, self.n_actions))
        self.n_episode_save = 5000
        self.logging = logging
        if self.logging:
            self.logger = Logger("/tmp/scalar/reward", "/tmp/scalar/avg-reward")

    def get_state_number(self, state):
        state_number = ((self.env.height*state[1]+state[0])*state[2])*state[3]
        if state_number <= self.n_states:
            return state_number
        else:
            return 0

    def epsilon_greedy(self, state):
        if self.epsilon > random.random():
            action = random.randint(0, self.n_actions-1)
        # Choose greedy action from Q table
        else :
            action = np.argmax(self.Q_table[self.get_state_number(state),:])
        return action

    def learn(self, n_episodes, n_steps):
        list_reward = []
        list_avg_reward= []
        total_reward = 0
        total_shoot_success = 0
        total_shoot_fail = 0
        for episode in range(n_episodes):
            state = self.env.reset()
            sum_reward = 0 
            if self.render and episode > 9000:
                self.env.render()
                time.sleep(0.1)

            self.epsilon = self.epsilon * self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

            for step in range(n_steps):
                action = self.epsilon_greedy(state)
                state_number = self.get_state_number(state)
                next_state, done, reward = self.env.step(action)
                next_state_number = self.get_state_number(next_state)
                current_Q = self.Q_table[state_number, action]
                if not done:
                    max_future_Q = np.max(self.Q_table[next_state_number,:])
                    Q_target = reward + self.gamma * max_future_Q
                else:
                    Q_target = reward
                self.Q_table[state_number, action] = current_Q + self.alpha * (Q_target - current_Q)
                
                state = next_state
                np.set_printoptions(precision=3)
                np.set_printoptions(suppress=True)
                sum_reward += reward
                if self.render and episode > 9000:
                    self.env.render()
                    time.sleep(0.1)
                if done:
                    break
            print("Episode :", episode, "Total Reward :", sum_reward)
            total_reward += sum_reward
            avg_reward = total_reward/(episode+1)
            if self.logging:
                self.logger.log_scalar("reward-per-episode", sum_reward, avg_reward, episode)
                list_reward.append(sum_reward)
                list_avg_reward.append(avg_reward)

            total_shoot_success += self.env.total_shoot_success
            total_shoot_fail += self.env.total_shoot_fail

        if self.logging:
            print("Total Shoot Success :", total_shoot_success)
            print("Total Shoot Fail :", total_shoot_fail)
            n = len(list_reward)
            t = np.arange(0,n,1)
            fig, ax = plt.subplots()
            ax.set_xlabel("Episode")
            ax.set_ylabel("Average Reward per Episode")
            ax.set_title("Average Reward VS Episode")
            ax.plot(t, list_avg_reward)
            plt.show()

class TDLambda:
    def __init__(self, alpha=0.01, gamma=0.8, lambd=0.9, epsilon=0.95, epsilon_decay=0.99, epsilon_min=0.01, render=False, logging=False):
        self.alpha = alpha 
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.epsilon_min = epsilon_min 
        self.epsilon_decay = epsilon_decay
        self.lambd = lambd
        self.render = render
        self.env = BasketballEnv(width=9, height=6, n_opponents=5, render=render, version=0)
        self.n_states = self.env.width * self.env.height * 2 * 2
        self.n_actions = len(self.env.action_dict)
        self.V = np.zeros(self.n_states)
        self.eligibility = np.zeros(self.n_states)
        self.logging = logging
        if self.logging:
            self.logger = Logger("/tmp/scalar/reward", "/tmp/scalar/avg-reward")

    def get_state_number(self, state):
        state_number = ((self.env.height*state[1]+state[0])*state[2])*state[3]
        if state_number <= self.n_states:
            return state_number
        else:
            return 0

    def epsilon_greedy(self, state):
        if self.epsilon > random.random():
            action = random.randint(0, self.n_actions-1)
        # Choose greedy action
        else :
            max_state_value = -999
            action = 0
            for i in range(self.n_actions):
                robot_x = self.env.robot_pos[0]
                robot_y = self.env.robot_pos[1]
                ball_x = self.env.ball_pos[0]
                ball_y = self.env.ball_pos[1]
                ns, _, _ = self.env.step(i)
                next_state_number = self.get_state_number(ns)
                if self.V[next_state_number] > max_state_value and next_state_number >= 0 and next_state_number < self.n_states:
                    max_state_value = self.V[next_state_number]
                    action = i
                self.env.robot_pos[0] = robot_x
                self.env.robot_pos[1] = robot_y
                self.env.ball_pos[0] = ball_x
                self.env.ball_pos[1] =  ball_y
        return action

    def learn(self, n_episodes, n_steps):
        total_reward = 0
        list_avg_reward = []
        list_reward = []
        total_shoot_success = 0
        total_shoot_fail = 0
        for episode in range(n_episodes):
            state = self.env.reset()
            sum_reward = 0 
            if self.render:
                self.env.render()

            self.epsilon = self.epsilon * self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

            for step in range(n_steps):
                action = self.epsilon_greedy(state)              
                next_state, done, reward = self.env.step(action)

                state_number = self.get_state_number(state)
                next_state_number = self.get_state_number(next_state)
                self.eligibility *= self.lambd * self.gamma
                self.eligibility[state_number] += 1.0
                td_error = reward + self.gamma * self.V[next_state_number] - self.V[state_number]
                self.V = self.V + self.alpha * td_error * self.eligibility

                state = next_state
                np.set_printoptions(precision=3)
                np.set_printoptions(suppress=True)
                sum_reward += reward
                if self.render:
                    self.env.render()
                if done:
                    break
            print("Episode :", episode, "Total Reward :", sum_reward)

            total_reward += sum_reward
            avg_reward = total_reward/(episode+1)
            if self.logging:
                self.logger.log_scalar("reward-per-episode", sum_reward, avg_reward, episode)
                list_reward.append(sum_reward)
                list_avg_reward.append(avg_reward)
            
            total_shoot_success += self.env.total_shoot_success
            total_shoot_fail += self.env.total_shoot_fail

        if self.logging:
            print("Total Shoot Success :", total_shoot_success)
            print("Total Shoot Fail :", total_shoot_fail)
            n = len(list_reward)
            t = np.arange(0,n,1)
            fig, ax = plt.subplots()
            ax.set_xlabel("Episode")
            ax.set_ylabel("Average Reward per Episode")
            ax.set_title("Average Reward VS Episode")
            ax.plot(t, list_avg_reward)
            plt.show()
            
def main():
    ql = QLearning(alpha=0.1, gamma=0.9, epsilon=0.95, epsilon_decay=0.99, epsilon_min=0.1, render=True, logging=False)
    ql.learn(10000, 50)
    
    # td = TDLambda(alpha=0.1, gamma=0.9, lambd=0.9, epsilon=0.95, epsilon_decay=0.99, epsilon_min=0.01, render=False, logging=True)
    # td.learn(10000, 50)
   
if __name__ == "__main__":
    main()