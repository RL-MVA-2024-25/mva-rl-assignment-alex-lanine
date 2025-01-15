#standard imports
import numpy as np 
import torch; import torch.nn as nn 
from gymnasium.wrappers import TimeLimit
import itertools 
import random 
import os 
from copy import deepcopy 

#local imports
from env_hiv_fast import FastHIVPatient as HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population
from utils import ReplayBuffer

#set up the environment
max_episode = 200
env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=max_episode
) 


#make projectagent 
class ProjectAgent:
    
    #initialize TO DO  
    def __init__(self, 
                 nb_actions = 4,
                 learning_rate = 1e-3,
                 gamma = 0.99,
                 batch_size = 1024, 
                 buffer_size = 1000000,
                 epsilon_max = 1.0, 
                 epsilon_min = 0.01,
                 epsilon_stop = 1000,
                 epsilon_delay = 100, 
                 criterion = torch.nn.SmoothL1Loss(),
                 nb_gradient_steps = 3,
                 update_target_freq = 400,
                 update_target_tau = 0.005
                ):
                 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model_dir = "./models/"
        self.nb_actions = nb_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.memory = ReplayBuffer(self.buffer_size,self.device)
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_stop = epsilon_stop
        self.epsilon_delay = epsilon_delay
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = self.defaul_model().to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.criterion = criterion
        self.lr = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.nb_gradient_steps = nb_gradient_steps
        self.update_target_strategy = 'replace'
        self.update_target_freq = update_target_freq
        self.update_target_tau = update_target_tau
        self.best_eval = -1e6

    print("test2")
        
    #take greedy dqn action
    def act(self, observation, use_random=False):
        #greedy action 
        if use_random: 
            return env.action_space.sample()
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(self.device))
        return torch.argmax(Q).item()
    

    def save(self, filename, subdir='best_model'):
        # Ensure the subdirectory exists
        save_path = os.path.join(self.model_dir, subdir)
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, filename)
        
        # Save the model
        torch.save(self.model.state_dict(), full_path)

    def load(self, filename, subdir='best_model'):
        load_path = os.path.join(self.model_dir, subdir, filename)       
        device = self.device
        self.model.load_state_dict(torch.load(load_path, map_location=device))
        self.model.eval()
        
        
    #define gradient step (from class)
    def gradient_step(self):
        if len(self.memory) < self.batch_size:
            return
        X, A, R, Y, D = self.memory.sample(self.batch_size)
        QYmax = self.target_model(Y).max(1)[0].detach()
        target = R + (1 - D) * self.gamma * QYmax
        QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
        loss = nn.SmoothL1Loss()(QXA, target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    #training function
    def train(self,env,max_episode,filename): 
        episode_return = []
        episode = 0
        step = 0 
        cum_reward = 0 
        epsilon = self.epsilon_max
        state, _ = env.reset()
        
        while episode < max_episode:
            #update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            #get action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else: 
                action = self.act(state)
            #take action 
            next_state, reward, done, trunc, _ = env.step(action)
            #save
            self.memory.append(state, action, reward, next_state, done)
            #update_reward
            cum_reward+=reward
            
            #start train loop 
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            
            #update network
            if step % self.update_target_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                
            #next step
            step+=1
            if done or trunc:
                episode+=1 
                eval_ = evaluate_HIV(agent=self,nb_episode=1)
                print(f"episode: {episode}; reward {cum_reward:.2e}; evaluation: {eval_}")
                if eval_ > self.best_eval:
                    self.best_model = deepcopy(self.model)
                    self.best_eval = eval_
                    self.save(filename=filename)
                episode_return.append(cum_reward)

                #reset
                state, _ = env.reset()
                cum_reward = 0 
                
            else:
                state = next_state
        
        self.model.load_state_dict(self.best_model.state_dict())
        self.save(filename=filename)
        return self.best_eval
        
    def default_model(self):
        device = self.device 
        state_dim = env.observation_space.shape[0]
        n_action = 4 
        nb_neurons = 200
        DQN = torch.nn.Sequential(nn.Linear(state_dim,nb_neurons),
                             nn.ReLU(),
                             nn.Linear(nb_neurons,nb_neurons),
                             nn.ReLU(),
                             nn.Linear(nb_neurons,nb_neurons),
                             nn.ReLU(),
                             nn.Linear(nb_neurons,nb_neurons),
                             nn.ReLU(),
                             nn.Linear(nb_neurons,n_action))
        
        return DQN
        

def main():
    #declare neteork
    print("test1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    n_action = 4
    nb_neurons= 512
    max_episode = 400    
    agent = ProjectAgent()
    agent.train(env, max_episode,filename="best_512model.pth")

                        
if __name__ == "__main__":
    main()