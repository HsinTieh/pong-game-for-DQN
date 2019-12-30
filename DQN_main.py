import random
import math
import os
import sys
import numpy as np
from itertools import count

import gym
from gym import wrappers

import torch
from torch import optim
import torch.autograd as autograd
import torch.nn.functional as F

from comm.wrappers import make_atari, wrap_deepmind
from agent.model import model
from comm.linear_schedule import LinearSchedule
from dqn_paramter import DQNparamter
from agent.replaybuffer import ReplayBuffer


def epsilon_decay(start=1.0, final=0.01, decay=30000):
    eps = lambda idx: final+(start-final) * math.exp(-1. * idx / decay)
    return eps

def selce_epilson_action(model, obs, t, DQN_paramter, action_num):
    r = random.random()
    threshold = DQN_paramter.exploration.value(t)
    if r > threshold:
        #依據 model 選擇action
        obs_ = torch.tensor(obs, dtype = torch.float).unsqueeze(0)
        if USE_CUDA:
            obs_ = obs_.cuda()
        q_value = model.forward(obs_)
       # print(q_value.max(1)[1].item() , type(q_value), type(q_value.max(1)[1].item() ))
        return q_value.max(1)[1].item()
     
    else:
        #隨機
        #tensor([[3]], dtype=torch.int32), <class 'torch.Tensor'>
        return torch.IntTensor([[random.randrange(action_num)]])

def load_model(name, model):
    if os.path.isfile(name):
        print('Loading model:', name)
        model.load_state_dict(torch.load(name))
    return model
def save_model(name, model):
    torch.save(model.state_dict(),name)
    print('saved model : ', name)

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
#Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

if __name__ == "__main__":
    SEED =0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # step 1 : build ENV for Pong game 
    env = make_atari('PongNoFrameskip-v0')
    env = wrap_deepmind(env)
    env = wrappers.Monitor(env, 'results/DQN', force = True, video_callable = lambda count: count%50 ==0)

    #step 2 : set DQN paramter
    eps = epsilon_decay()
    schedule = LinearSchedule()   
    paramter = DQNparamter(exploration = schedule, \
                            replay_buffer_size = 1000000, \
                            batch_size = 32, \
                            gamma = 0.99, \
                            learning_start = 50000, \
                            learning_ferg = 4, \
                            frame_history_len = 4, \
                            target_updata_freq = 1000, \
                            eps = eps, \
                            learning_rate = 0.00025, \
                            alpha = 0.95)

    h,w,c = env.observation_space.shape
    #依據四個frame作為輸入
    input_arg = paramter.frame_history_len*c  
    action_num = env.action_space.n
    print('env:', h,w,c,input_arg)
    print('action : ', env.unwrapped.get_action_meanings(), action_num)

    #step 3 : build Q  and target Q
    Q       = model(input_arg,action_num)
    targetQ = model(input_arg,action_num)
    # 檢查是否有已訓練過的pkl檔案
    Q = load_model('training model pkl/DQN/Q_parame.pkl', Q)
    targetQ = load_model('training model pkl/DQN/target_Q_parame.pkl', targetQ)
    if USE_CUDA:
        Q.cuda()
        targetQ.cuda()
    #step 4 : build replaybuffer
    buffer = ReplayBuffer(paramter.replay_buffer_size, paramter.frame_history_len)

    #step 5 : build optimizer
    optimizer = optim.Adam(Q.parameters(), lr=paramter.learning_rate)  

   
    target_network_counter = 0 #targetQ 更新之計數器
    print_log_freq = 20000 #列印log and save model的頻率
    best_mean_episode_reward = -float('inf')
    obs = env.reset()

    #step 6 : start training model
    for t in count():
        #存入obs_ 狀態並取的 obs index
        #e.g. t=1 ----->(s,_,_,_,_)  。 t=2 ----->(s',_,_,_,_)
        obs_index = buffer.push_obs(obs)
        #取得存放的obsssss，具有四個frame的意思
        obs_many = buffer.pull_obs(buffer.next_index -1 % buffer.capacity, 'now')
        
        
        #一開始先隨機，等收集到一定量再做訓練
        if t < paramter.learning_start:
            #隨機
            action = random.randrange(action_num)
        else:
            #訓練
            action = selce_epilson_action(Q, obs_many, t, paramter, action_num)
           # print(action)
        #獲取資訊
        obs_, reward, done, info = env.step(action)
        

        #評估reward ------> if r > 1 => r=1 ::::: if r < -1.0 => r= -1
        reward = max(-1.0, min(reward,1.0))
        #儲存新的資訊
        #e.g. t=1 -----> (s,a,r,s',done)。 t=2 ----->(s',a',r',s'',done')
        buffer.push(obs_index, action, reward, obs_, done)
        
        #檢查遊戲是否結束
        if done:
            obs_ = env.reset()
        # s ----> s'
        obs = obs_
        if t%1000 == 0:
            print('Timestep:',t)
        ''' 訓練network的部分'''
        if t > paramter.learning_start and \
            t % paramter.learning_ferg == 0 and \
            buffer.check_sample(paramter.batch_size) :

            #取樣囉 ~~~~~~~~ ^____^
            obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = buffer.pull_for_sample(paramter.batch_size)
            #print('obs shape:',obs_batch.shape, next_obs_batch.shape)

            #print(type(obs_batch[0]), type(action_batch[0]), type(reward_batch[0]), type(next_obs_batch[0]))
            obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype) / 255.0)
            action_batch = Variable(torch.from_numpy(action_batch).long())
            reward_batch = Variable(torch.from_numpy(reward_batch))
            next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype) / 255.0)
            not_done_batch = Variable(torch.from_numpy(1 - done_batch)).type(dtype)

            if USE_CUDA:
                obs_batch = obs_batch.cuda()
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()
                next_obs_batch = next_obs_batch.cuda()
                not_done_batch = not_done_batch.cuda()
                
 
        
            '''
            torch.gather(input, dim, index, out=None)
            dim = 0 ---->直向
            dim = 1 ---->橫向
            index ------>列號
            e.g.
            |1 2 3|       1.torch.gather(input, 1, torch.LongTensor[[0,1],[2,0]])
            |4 5 6|       2.torch.gather(input, 0, torch.LongTensor[[0,1,1],[0,0,0]])
            1.          2.
            |1 2|       |1 5 6|
            |6 4|       |1 2 3|

            torch.squeeze(input, dim, out=None)   #默认移除所有size为1的维度，当dim指定时，移除指定size为1的维度. 返回的tensor会和input共享存储空间，所以任何一个的改变都会影响另一个
            torch.unsqueeze(input, dim, out=None) #扩展input的size, 如 A x B 变为 1 x A x B 
            
            '''

            '''
            :::DQN:::
            Q(s,a) <-------> r + maxQ'(s',a)

            '''

            #從抽出的batch observation中得出現在的Q值
            q_value = Q(obs_batch)
            current_Q_values = q_value.gather(1, action_batch.unsqueeze(1)).squeeze(1)
            #用next_obs_batch計算下一個Q值，detach代表將target network從graph中分離，不去計算它的gradient
            next_Q_value = targetQ(next_obs_batch).detach().max(1)[0]

            #TD value
            target_Q_value = reward_batch + (paramter.gamma * next_Q_value)
            
            loss = F.smooth_l1_loss(current_Q_values, target_Q_value)
            
            #backward & update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            target_network_counter +=1
            #每隔一段時間更新 target network
            if target_network_counter % paramter.target_updata_freq == 0:
                targetQ.load_state_dict(Q.state_dict())
                              
            #log message
            if t % print_log_freq == 0 and t > paramter.learning_ferg:
                episode_rewards = env.get_episode_rewards()
                if len(episode_rewards) > 0:
                    mean_episode_reward = np.mean(episode_rewards[-100:])
                    if len(episode_rewards) > 10:
                        print('enter best_mean_episode_reward')
                        best_mean_episode_reward = max(best_mean_episode_reward,mean_episode_reward)
            
                print("Timestep : {:d} \nmean reward : {:f} \nbest mean reward : {:f} \nepisodes : {:d}\n\n" \
                    .format(t,mean_episode_reward,best_mean_episode_reward,len(episode_rewards)))
                
                sys.stdout.flush()
                # save model
                save_model('training model pkl/DQN/Q_params.pkl', Q)
                save_model('training model pkl/DQN/target_Q_parame.pkl',targetQ)               






    


