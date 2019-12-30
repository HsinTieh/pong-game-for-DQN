class DQNparamter():
    def __init__(self, exploration, replay_buffer_size, batch_size, gamma, learning_start, learning_ferg, frame_history_len, target_updata_freq, learning_rate,alpha, eps):
        self.exploration = exploration
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_start = learning_start
        self.learning_ferg = learning_ferg
        self.frame_history_len = frame_history_len
        self.target_updata_freq = target_updata_freq
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.eps = eps
    