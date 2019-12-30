import numpy as np
import random
class ReplayBuffer(object):
    def __init__(self, capacity, frame_history_len):
        
        self.capacity = capacity
        
        self.current_buffer_coutter = 0 # 目前buffer的儲量
        self.next_index = 0 # s'
        self.frame_history_len = frame_history_len 
        
        #建立buffer ，最大容量為capacity
        self.obs = None
        self.reward = None
        self.action = None
        self.next_obs = None
        self.done = None

        # for multi_step 
        self.n_step_buffer = []
    #儲存狀態    
    def push_obs(self, obs):
        if len(obs.shape) >1 :
            #transpose image obs(frame) into (c,h,w)
            obs = obs.transpose(2,0,1)
            
        #給於變數定義
        if self.obs is None:
            self.obs = np.empty([self.capacity]+list(obs.shape), dtype = np.uint8)
            self.reward = np.empty([self.capacity], dtype = np.float32)
            self.action = np.empty([self.capacity], dtype = np.int32)
            self.next_obs = np.empty([self.capacity]+list(obs.shape), dtype = np.uint8)
            self.done = np.empty([self.capacity], dtype = np.bool)
            print('buffer build .......ok')
        #儲存新的obs資訊
        self.obs[self.next_index] = obs

        index = self.next_index
        
        #移到下一個位置，如果超過buffer就重新覆蓋
        self.next_index = (self.next_index+1) % self.capacity
        
        #buffer儲量數加1
        self.current_buffer_coutter = min(self.capacity, self.current_buffer_coutter+1)
    
        return index       
    
    #採樣buffer資訊    
    def pull_obs(self, index, obs_type):
        #因為取四個frame，所以要往回推四個
        index_end = index +1
        index_start = index_end - self.frame_history_len
        if obs_type == 'now':
            obs_ = self.obs
        else:
            obs_ =self.next_obs
        # check if it is low-dimensional obs, such as RAM
        if len(obs_.shape) == 2: return obs_[index_end-1]
        
        #檢查是否有足夠的obs(frame)
        if index_start < 0 and self.current_buffer_coutter != self.capacity:
            index_start = 0
            
        #檢查 連續的obs 是否已完成遊戲，有就忽略那格frame
        for idx in range(index_start, index_end -1):
            if self.done[idx%self.capacity]:
                index_start = idx+1
    
        #計算有多少缺失的obs ----> <0表示沒有足夠obs,>0表示有被忽略的frame
        miss_obs = self.frame_history_len - (index_end - index_start)
        # 檢查是否有足夠的obs or 檢查是否存在已經完成遊戲obs
        if miss_obs < 0 or miss_obs > 0:

            #用第一個obs的補齊缺少的部分
            frames = [np.zeros_like(obs_[0]) for _ in range(miss_obs)]
            #再將原本有的obs加入 frames
            for idx in range(index_start,index_end):
                frames.append(obs_[idx % self.capacity])
            #封裝回傳
            r = np.concatenate(frames, 0)

            return r
        else:
            h,w = obs_.shape[2], obs_.shape[3]
            if index_start < 0:
                index_end = index_end - index_start
                index_start = 0
            r = obs_[index_start:index_end].reshape(-1, h,w)
           
            return r 
    #儲存新的資訊
    def push(self, index, action, reward, next_state, done):
        self.action[index] = action
        self.reward[index] = reward
        if len(next_state.shape) >1 :
            #transpose image obs(frame) into (c,h,w)
            next_state = next_state.transpose(2,0,1)
        self.next_obs[index] = next_state
        self.done[index] = done

    #檢查是否有足夠的數量能採樣
    def check_sample(self, batch_size):
        return batch_size + 1 <= self.current_buffer_coutter
    

    # 採樣buffer的資訊  
    def pull_for_sample(self, batch_size):
        
        assert self.check_sample(batch_size)

        #sampling
        idxes = [random.randint(0, self.current_buffer_coutter - 1) for _ in range(batch_size)]
            
        obs_batch      = np.concatenate([self.pull_obs(idx,'now')[np.newaxis, :] for idx in idxes], 0)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = np.concatenate([self.pull_obs(idx,'next')[np.newaxis, :] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32) # 將True, False轉為1, 0
        
        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask

    def multi_step(self, a,r,s_ ,d,n_step, gamma):
        self.n_step_buffer.append((a,r,s_,d))
        if n_step > len(self.n_step_buffer):
            #不讓buffer移到下一個index
            self.next_index -=1
            return
        
        R = sum([self.n_step_buffer[i][2] * (gamma**i) for i in range(n_step)])
        action, reward, state_ ,done = self.n_step_buffer.pop(0)
        print(R)
        self.push(self.next_index-1, action, R, s_, d )
