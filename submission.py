#### rps_stpete_ishii_160th_place_solution_20210225 ####

import pandas as pd
import numpy as np
import json
import random
from random import randrange
import collections


class agent():
    def initial_step(self):
        return np.random.randint(3)
    
    def history_step(self, history):
        return np.random.randint(3)
    
    def step(self, history):
        if len(history) == 0:
            return int(self.initial_step())
        else:
            return int(self.history_step(history))

         
############## additional agents  ##############
class rps(agent):
    def __init__(self, shift=0):
        self.shift = shift
    
    def rps(self, history):
        return self.shift % 3

      
############## origial agents added ##############
class j8(agent):

    def my_agent_j8(self, observation, configuration):

        global opp_history, T, P
        opp_history=[]    
        T = np.ones((3, 3, 3, 3, 3, 3, 3, 3, 3))
        P = np.zeros((3, 3))    
        a1, a2 = None, None

        if observation['step'] == 0 :
            next_hand = np.random.choice( 3 )   
            return next_hand   

        elif observation['step'] < 10 :
            opp_history.append(observation["lastOpponentAction"])   
            next_hand = np.random.choice( 3 )   
            return next_hand  

        else:
            opp_history.append(observation["lastOpponentAction"])   

            ai1 = opp_history[-1]
            ai2 = opp_history[-2]
            ai3 = opp_history[-3]
            ai4 = opp_history[-4] 
            ai5 = opp_history[-5] 
            ai6 = opp_history[-6]  
            ai7 = opp_history[-7]  
            ai8 = opp_history[-8] 
            ai9 = opp_history[-9] 

            T[ai9,ai8,ai7,ai6,ai5,ai4,ai3,ai2,ai1] += 1

            a1 = opp_history[-1]
            a2 = opp_history[-2]
            a3 = opp_history[-3]
            a4 = opp_history[-4]  
            a5 = opp_history[-5]         
            a6 = opp_history[-6]  
            a7 = opp_history[-7] 
            a8 = opp_history[-8] 

            T1 = T[a8,a7,a6,a5,a4,a3,a2]

            P = np.divide(T1, np.maximum(1, T1.sum(axis=1)).reshape(-1, 1))   
            next_hand = (np.random.choice(3, p=P[a1,:]) +1) %3    

            return next_hand

        
############## origial agents added ##############
class hx(agent):
    def __init__(self, shift=0):
        self.shift = shift
        
    def my_agent_hx(self, observation, configuration):
        global opp_history
        opp_history=[]

        if observation['step'] == 0  :
            return np.random.choice( 3 )   

        elif observation['step'] < self.shift+1 :
            opp_history.append(observation["lastOpponentAction"])   
            c = collections.Counter(opp_history)
            n = c[0] + c[1] + c[2]
            max0 = max(c[0],c[1],c[2])
            min0 = max(c[0],c[1],c[2])

            if max0 == c[0] and min0 == c[1]:
                next_hand = np.random.choice( 3, p=[c[2]/(2*n),c[0]/n+c[1]/n,c[2]/(2*n)] )   
            elif max0 == c[0] and min0 == c[2]:
                next_hand = np.random.choice( 3, p=[c[1]/(2*n),c[0]/n+c[2]/n,c[1]/(2*n)] )           
            elif max0 == c[1] and min0 == c[0]:
                next_hand = np.random.choice( 3, p=[c[2]/(2*n),c[2]/(2*n),c[1]/n+c[0]/n] )          
            elif max0 == c[1] and min0 == c[2]:
                next_hand = np.random.choice( 3, p=[c[0]/(2*n),c[0]/(2*n),c[1]/n+c[2]/n] )         
            elif max0 == c[2] and min0 == c[1]:
                next_hand = np.random.choice( 3, p=[c[2]/n+c[1]/n,c[0]/(2*n),c[0]/(2*n)] )         
            elif max0 == c[2] and min0 == c[0]:
                next_hand = np.random.choice( 3, p=[c[2]/n+c[0]/n,c[1]/(2*n),c[1]/(2*n)] ) 
            else:
                next_hand = np.random.choice( 3 )             

            return next_hand     

        else:
            opp_history.append(observation["lastOpponentAction"])   
            c = collections.Counter(opp_history[-self.shift:])
            n = c[0] + c[1] + c[2]
            max0 = max(c[0],c[1],c[2])
            min0 = max(c[0],c[1],c[2])

            if max0 == c[0] and min0 == c[1]:
                next_hand = np.random.choice( 3, p=[c[2]/(2*n),c[0]/n+c[1]/n,c[2]/(2*n)] )   
            elif max0 == c[0] and min0 == c[2]:
                next_hand = np.random.choice( 3, p=[c[1]/(2*n),c[0]/n+c[2]/n,c[1]/(2*n)] )           
            elif max0 == c[1] and min0 == c[0]:
                next_hand = np.random.choice( 3, p=[c[2]/(2*n),c[2]/(2*n),c[1]/n+c[0]/n] )          
            elif max0 == c[1] and min0 == c[2]:
                next_hand = np.random.choice( 3, p=[c[0]/(2*n),c[0]/(2*n),c[1]/n+c[2]/n] )         
            elif max0 == c[2] and min0 == c[1]:
                next_hand = np.random.choice( 3, p=[c[2]/n+c[1]/n,c[0]/(2*n),c[0]/(2*n)] )         
            elif max0 == c[2] and min0 == c[0]:
                next_hand = np.random.choice( 3, p=[c[2]/n+c[0]/n,c[1]/(2*n),c[1]/(2*n)] ) 
            else:
                next_hand = np.random.choice( 3 )             

            return next_hand 


    

class mirror_shift(agent):
    def __init__(self, shift=0):
        self.shift = shift
    
    def history_step(self, history):
        return (history[-1]['competitorStep'] + self.shift) % 3
    

class self_shift(agent):
    def __init__(self, shift=0):
        self.shift = shift
    
    def history_step(self, history):
        return (history[-1]['step'] + self.shift) % 3    


class popular_beater(agent):
    def history_step(self, history):
        counts = np.bincount([x['competitorStep'] for x in history])
        return (int(np.argmax(counts)) + 1) % 3

    
class anti_popular_beater(agent):
    def history_step(self, history):
        counts = np.bincount([x['step'] for x in history])
        return (int(np.argmax(counts)) + 2) % 3
    
    
class transition_matrix(agent):
    def __init__(self, deterministic = False, counter_strategy = False, init_value = 0.1, decay = 1):
        self.deterministic = deterministic
        self.counter_strategy = counter_strategy
        if counter_strategy:
            self.step_type = 'step' 
        else:
            self.step_type = 'competitorStep'
        self.init_value = init_value
        self.decay = decay
        
    def history_step(self, history):
        matrix = np.zeros((3,3)) + self.init_value
        for i in range(len(history) - 1):
            matrix = (matrix - self.init_value) / self.decay + self.init_value
            matrix[int(history[i][self.step_type]), int(history[i+1][self.step_type])] += 1

        if  self.deterministic:
            step = np.argmax(matrix[int(history[-1][self.step_type])])
        else:
            step = np.random.choice([0,1,2], p = matrix[int(history[-1][self.step_type])]/matrix[int(history[-1][self.step_type])].sum())
        
        if self.counter_strategy:
            return (step + 2) % 3 
        else:
            return (step + 1) % 3
    

class transition_tensor(agent):
    
    def __init__(self, deterministic = False, counter_strategy = False, init_value = 0.1, decay = 1):
        self.deterministic = deterministic
        self.counter_strategy = counter_strategy
        if counter_strategy:
            self.step_type1 = 'step' 
            self.step_type2 = 'competitorStep'
        else:
            self.step_type2 = 'step' 
            self.step_type1 = 'competitorStep'
        self.init_value = init_value
        self.decay = decay
        
    def history_step(self, history):
        matrix = np.zeros((3,3,3)) + 0.1
        for i in range(len(history) - 1):
            matrix = (matrix - self.init_value) / self.decay + self.init_value
            matrix[int(history[i][self.step_type1]), int(history[i][self.step_type2]), int(history[i+1][self.step_type1])] += 1

        if  self.deterministic:
            step = np.argmax(matrix[int(history[-1][self.step_type1]), int(history[-1][self.step_type2])])
        else:
            step = np.random.choice([0,1,2], p = matrix[int(history[-1][self.step_type1]), int(history[-1][self.step_type2])]/matrix[int(history[-1][self.step_type1]), int(history[-1][self.step_type2])].sum())
        
        if self.counter_strategy:
            return (step + 2) % 3 
        else:
            return (step + 1) % 3


class pattern_matching(agent):
    def __init__(self, steps = 3, deterministic = False, counter_strategy = False, init_value = 0.1, decay = 1):
        self.deterministic = deterministic
        self.counter_strategy = counter_strategy
        if counter_strategy:
            self.step_type = 'step' 
        else:
            self.step_type = 'competitorStep'
        self.init_value = init_value
        self.decay = decay
        self.steps = steps
        
    def history_step(self, history):
        if len(history) < self.steps + 1:
            return self.initial_step()
        
        next_step_count = np.zeros(3) + self.init_value
        pattern = [history[i][self.step_type] for i in range(- self.steps, 0)]
        
        for i in range(len(history) - self.steps):
            next_step_count = (next_step_count - self.init_value)/self.decay + self.init_value
            current_pattern = [history[j][self.step_type] for j in range(i, i + self.steps)]
            if np.sum([pattern[j] == current_pattern[j] for j in range(self.steps)]) == self.steps:
                next_step_count[history[i + self.steps][self.step_type]] += 1
        
        if next_step_count.max() == self.init_value:
            return self.initial_step()
        
        if  self.deterministic:
            step = np.argmax(next_step_count)
        else:
            step = np.random.choice([0,1,2], p = next_step_count/next_step_count.sum())
        
        if self.counter_strategy:
            return (step + 2) % 3 
        else:
            return (step + 1) % 3

        
agents = {
       
# origial agents added
    'j8': j8(),        
    'hx200': hx(200),   
    'hx250': hx(250),

# additional agents    
    'rps_0': rps(0),
    'rps_1': rps(1),
    'rps_2': rps(2),
        
    'mirror_0': mirror_shift(0),
    'mirror_1': mirror_shift(1),  
    'mirror_2': mirror_shift(2),
    
    'self_0': self_shift(0),
    'self_1': self_shift(1),  
    'self_2': self_shift(2),
    
    'popular_beater': popular_beater(),
    'anti_popular_beater': anti_popular_beater(),
    'random_transitison_matrix': transition_matrix(False, False),
    'determenistic_transitison_matrix': transition_matrix(True, False),
    'random_self_trans_matrix': transition_matrix(False, True),
    'determenistic_self_trans_matrix': transition_matrix(True, True),
    'random_transitison_tensor': transition_tensor(False, False),
    'determenistic_transitison_tensor': transition_tensor(True, False),
    'random_self_trans_tensor': transition_tensor(False, True),
    'determenistic_self_trans_tensor': transition_tensor(True, True),
    
    'random_transitison_matrix_decay': transition_matrix(False, False, decay = 1.05),
    'random_self_trans_matrix_decay': transition_matrix(False, True, decay = 1.05),
    'random_transitison_tensor_decay': transition_tensor(False, False, decay = 1.05),
    'random_self_trans_tensor_decay': transition_tensor(False, True, decay = 1.05),
    
     'random_pattern_matching_decay_5': pattern_matching(5, False, False, decay = 1.001),
     'random_self_pattern_matching_decay_5': pattern_matching(5, False, True, decay = 1.001),
     'determenistic_pattern_matching_decay_5': pattern_matching(5, True, False, decay = 1.001),
     'determenistic_self_pattern_matching_decay_5': pattern_matching(5, True, True, decay = 1.001),
    
     'random_pattern_matching_decay_6': pattern_matching(6, False, False, decay = 1.001),
     'random_self_pattern_matching_decay_6': pattern_matching(6, False, True, decay = 1.001),
     'determenistic_pattern_matching_decay_6': pattern_matching(6, True, False, decay = 1.001),
     'determenistic_self_pattern_matching_decay_6': pattern_matching(6, True, True, decay = 1.001),
}

history = []
bandit_state = {k:[1,1] for k in agents.keys()}
    
def multi_armed_bandit_agent (observation, configuration):
    
    # bandits' params
    step_size = 3 
    decay_rate = 1.1
    
    global history, bandit_state
    
    def log_step(step = None, history = None, agent = None, competitorStep = None, file = 'history.csv'):
        if step is None:
            step = np.random.randint(3)
        if history is None:
            history = []
        history.append({'step': step, 'competitorStep': competitorStep, 'agent': agent})
        if file is not None:
            pd.DataFrame(history).to_csv(file, index = False)
        return step
    
    def update_competitor_step(history, competitorStep):
        history[-1]['competitorStep'] = int(competitorStep)
        return history
    
    if observation.step == 0:
        pass
    else:
        history = update_competitor_step(history, observation.lastOpponentAction)
        
        for name, agent in agents.items():
            agent_step = agent.step(history[:-1])
            bandit_state[name][1] = (bandit_state[name][1] - 1) / decay_rate + 1
            bandit_state[name][0] = (bandit_state[name][0] - 1) / decay_rate + 1
            
            if (history[-1]['competitorStep'] - agent_step) % 3 == 1:
                bandit_state[name][1] += step_size
            elif (history[-1]['competitorStep'] - agent_step) % 3 == 2:
                bandit_state[name][0] += step_size
            else:
                bandit_state[name][0] += step_size/2
                bandit_state[name][1] += step_size/2
            
    with open('bandit.json', 'w') as outfile:
        json.dump(bandit_state, outfile)
    
    best_proba = -1
    best_agent = None
    
    factor = 0.2  ##### set factor value
    
    for k in bandit_state.keys():
        proba = np.random.beta(factor*bandit_state[k][0],factor*bandit_state[k][1])    ##### set factor
        if proba > best_proba:
            best_proba = proba
            best_agent = k
        
    step = agents[best_agent].step(history)
    
    return log_step(step, history, best_agent)
