import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor, LongTensor, optim
from functools import reduce 
import sys
# import matplotlib.pyplot as plt
from parameter_set import variants
import pickle

# from parameters import *

# sys.path.append('/home/elliott_lab/shared/github_paramtest/mixture_of_experts_RL/')

'''
# ALSO WORKS:
parser = argparse.ArgumentParser()
parser.add_argument("--argfile")
args = parser.parse_args()
param1 = importlib.import_module(args.argfile, ".")
'''
#parser = argparse.ArgumentParser()
#parser.add_argument(sys.argv[1])
#args = parser.parse_args()

# ALSO WORKS: 
# specific_parameters = sys.argv[1]
# general_parameter_location = '/home/elliott_lab/shared/github_paramtest/mixture_of_experts_RL/single_run_data'
# specific_parameter_location = os.path.join(general_parameter_location, specific_parameters)
# sys.path.insert(1, specific_parameter_location)



'''
# CURRENT THING THAT WORKS- READING FROM A PYTHON SCRIPT (IMPORTED AS MODULE): 
param1 = importlib.import_module(sys.argv[1], ".")

# argv=param1.v
# argN_e=param1.N_e
# argscgI=param1.scgI
arggamma=param1.gamma
argM_H=param1.M_H
argnumReplays=param1.numReplays
argbatchSize=param1.batchSize
argnumEpisodes=param1.numEpisodes
argepisodeLen=param1.episodeLen
arglr=param1.lr
argmomentum=param1.momentum
argrender=param1.render
'''

'''
# ORIGINAL PARAMETERS

parser = argparse.ArgumentParser()
parser.add_argument("-v", action="store_true", default=False, help="not implemented")
parser.add_argument("--M_H", nargs="+", type=int, default=[10,10])
parser.add_argument("--N_e", type=int, default=1)
parser.add_argument("--scgI", type=int, default=20)
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--rerunNum", type=int)
parser.add_argument("--numReplays", type=int, default=50)
parser.add_argument("--batchSize", type=int, default=50)
parser.add_argument("--numEpisodes", type=int, default=5000)
parser.add_argument("--episodeLen", type=int, default=250)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.2)
parser.add_argument("--render", action="store_true", default=False)
# parser.add_argument("--graph",action="store_true", default=False)
# parser.add_argument("--saveEvalHist",action="store_true", default=False)
# parser.add_argument("--saveWeightsInterval", type=int, default=0, help="how often to save weights.  zero for don't save. Value of one will save every eval.")
# parser.add_argument("--saveW",action="store_true", default=False)
# parser.add_argument("--termEvalR", nargs="+", type=int, default=None)
parser.add_argument("--saveDir")
parser.add_argument("--savePrefix")
# parser.add_argument("--initWeights", nargs="+", default=None, help="File from which to load initial weight values.")
args = parser.parse_args()
print(args)
'''

index = sys.argv[1]
index = int(index)

argM_H=(variants[index][0])
argnumReplays=variants[index][1]
argbatchSize=variants[index][2]
arglr=variants[index][3]
argmomentum=variants[index][4]

arggamma=0.9
argnumEpisodes=3001
argepisodeLen=250


#
# define a class for the ANN representing the Q-function
# instantiate the class
class Net(nn.Module):
    def __init__(self , M_I , M_H):
        super(Net, self).__init__()
        self.M_H = M_H
        self.Qlayers = []
        # create hidden layers
        self.h0 = nn.Linear(M_I,self.M_H[0])
        for li in range(1,len(self.M_H)):
            attrName = "h"+str(li)
            setattr( self, attrName , nn.Linear(self.M_H[li-1],self.M_H[li]) )
            self.Qlayers += [attrName]
        # create output layer with single output
        self.qout = nn.Linear(self.M_H[-1],1)
        self.Qlayers += ["qout"]
        
    def forward(self, x):
        for li in range(len(self.M_H)):
            layer = getattr(self,"h"+str(li))
            x = F.tanh(layer(x))
        x = self.qout(x)
        return x

    def select_action(self,s,a_all):
        with torch.no_grad():
            Qvals = []
            for a in a_all:
                Qin = Variable(Tensor(np.append(s , a)))
                Qout = self.forward(Qin).data
                Qvals += [Qout[0]]
        return np.argmax(Qvals)

    def get_parameters(self):
        return iter(reduce(lambda x,y: x+y, [list(mod.parameters()) for name,mod in self.named_children() if name in self.Qlayers]))

def evaluate(agent, startoff, agent_action):
    eval_state = env.modreset(startoff)
    eval_reward_sum = 0
    for _ in range(argepisodeLen):
        act_like_this = agent.select_action(eval_state, agent_action)
        a_eval = agent_action[act_like_this:act_like_this+1]
        eval_state_next, eval_reward, eval_done, eval_info = env.step(a_eval)
        eval_reward_sum = eval_reward_sum + eval_reward
        eval_state = eval_state_next
    return eval_reward_sum

x_axis = []
angle_y_axis = []
reward_y_axis = []
eval_avg_reward = []
filename = sys.argv[1]
# out = open(filename, 'a')

Qfunc = Net(4, argM_H)
# done
#

#
# create available actions
actions = np.linspace(-2. , 2. , 10)
# done
#

#
# epsilon-greedy setup
# if args.epsilon_start > 0:
#     epsilon = args.epsilon
# else:
#     epsilon = 0.1
# minEpsilon = 0.1
# epsilonDecay = np.exp(np.log(minEpsilon)/float(T_max))
# epsilon = 0
epsilon = 1.0
# done
# 

#
# create training environment
env = gym.make('Pendulum-v0')
# done
#

#
# initialize PyTorch optimization algorithm
# optimizer = optim.Adam(Qfunc.parameters(), lr=args.lr, weight_decay=0.0) 
optimizer = optim.SGD(Qfunc.parameters(), lr=arglr, momentum=argmomentum) 
# done
# 

#
# loop over episodes
#
memory = []
for episode_i in range(argnumEpisodes):
    #
    # reset the environment
    state = env.reset()
    # done
    #

    #
    # loop over episode time steps
    for t in range(argepisodeLen):
        #
        # optionally draw the env
        '''
        NOT RENDERING AT THE MOMENT, SO:
        if argrender:
            env.render()
        '''
        # done
        #

        #
        # select an action
        if np.random.uniform() > epsilon:
            a_i = random.randint(0,len(actions)-1)
        else:
            a_i = Qfunc.select_action(state , actions)
        a = actions[a_i:a_i+1]  # need a single-element list
        # done
        #

        #
        # TODO: add epsilon-decay
        # done
        # 

        #
        # advance to the next state by taking selected action
        s_next , r , done , info = env.step(a)
        # done
        #

        #
        # add experience to replay memory
        # make next state the current state
        memory += [(state , a_i , r , s_next)]
        state = s_next
        # done
        # 
        
    # end loop over interactions with environment
    #

    #
    # loop over the number of replays for each parameter update session
    for replay_i in range(argnumReplays):
        # 
        # select training samples from replay memory
        # if not enough memories, just select some fraction of them
        selected_memories_idxs = np.random.randint(0 , len(memory) ,
                                                   min(int(len(memory)*0.5),argbatchSize))
        training_inputs = Tensor(np.vstack([memory[smi][0] for smi in selected_memories_idxs])) # batchSize x M_I-1
        # done
        #

        #
        # create training data target values
        next_states = Tensor(np.vstack([memory[smi][3] for smi in selected_memories_idxs]))
        qvals_next_tmp = torch.zeros((len(selected_memories_idxs),len(actions)))
        with torch.no_grad():
            for ai in range(len(actions)):
                # compute the Q-values for max Q(s(t+1),a) for all a in actions
                X = torch.cat((next_states ,
                               Tensor([[actions[ai] for i in range(training_inputs.shape[0])]]).t()) ,
                              1)
                qvals_next_tmp[:,ai:ai+1] = Qfunc.forward(Variable(X)).data
        X = torch.cat((training_inputs ,
                       Tensor([[actions[memory[smi][1]] for smi in selected_memories_idxs]]).t()) ,
                      1)
        Q = Qfunc.forward(Variable(X))

        rewards = Tensor([memory[smi][2] for smi in selected_memories_idxs]).unsqueeze(1)
        qvals_next , selected_actions_next = torch.max(qvals_next_tmp,dim=1)
        qvals_next = qvals_next.unsqueeze(1)
        T = rewards + arggamma * qvals_next
        # done
        # 

        # 
        # update model
        optimizer.zero_grad()
        loss = F.mse_loss(Q , T)
        loss.backward()
        optimizer.step()
        # done
        #

    # end of loop over replays
    # 

    #
    # print out performance on that episode
    rewardSum = np.sum([mem[2] for mem in memory[-argepisodeLen:]])
    print("Total reward for episode ", episode_i ,":", rewardSum)
    reward_y_axis.append(rewardSum)
    # done
    # 
    
    if ((episode_i) % 100 == 0):
        state_list = [[-np.pi, -1], [-np.pi, -.5], [-np.pi, .5], [-np.pi, 1], [-np.pi/2, -1], [-np.pi/2, -.5],\
                      [-np.pi/2, .5], [-np.pi/2, 1], [np.pi/2, -1], [np.pi/2, -.5], [np.pi/2, .5],\
                      [np.pi/2, 1], [np.pi, -1], [np.pi, -.5], [np.pi, .5], [np.pi, 1]]
        total_eval_rwd = 0
        for st in state_list:
            sumrwd = evaluate(Qfunc, st, actions)
            total_eval_rwd += sumrwd
    
        eval_avg_reward.append(total_eval_rwd/16)
    
    
# out.close()  
'''
xpicklefile = "x"+str(filename)+".pickle"
x_pickle = open(xpicklefile, "wb")
pickle.dump(x_axis, x_pickle)
x_pickle.close()
'''
eval_picklefile = "eval"+str(filename)+".pickle"
eval_pickle = open(eval_picklefile, "wb")
pickle.dump(eval_avg_reward, eval_pickle)
eval_pickle.close()

ypicklefile = "y"+str(filename)+".pickle"
y_pickle = open(ypicklefile, "wb")
pickle.dump(reward_y_axis, y_pickle)
y_pickle.close()

# end of loop over training episodes
# 
#PICKLE STUFFS:

#CURRENT WORKING PLOT:
'''    
fig, ax = plt.subplots(figsize = (17,9))
ax.plot(x_axis, reward_y_axis, 'g')
ax.set(xlabel='Episode', ylabel='Reward', title='Episode vs Reward')
ax.grid()
pngname = str(sys.argv[1])
fig.savefig(pngname)
plt.show()
'''
 
'''
# perfect side-by-side 2 plot script follows, look no further! 
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,7), dpi=120)
ax1.plot(x_axis, angle_y_axis, 'g')
ax2.plot(x_axis, reward_y_axis, 'm')
ax1.set_title('Episode vs Angle'); ax2.set_title('Episode vs Reward')
ax1.set_xlabel('Episodes');  ax2.set_xlabel('Episodes')  
ax1.set_ylabel('Angle');  ax2.set_ylabel('Reward')  
plt.grid()
plt.tight_layout()
plt.show()
pngname = str(sys.argv[1])
fig.savefig(pngname)
'''