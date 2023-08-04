# Import pakages
import torch
import torch.nn as nn
import gym
import os
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
# Import python files
from utils import v_wrap, set_init, push_and_pull, record
from shared_adam import SharedAdam

os.environ["OMP_NUM_THREADS"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Setting hyperparameters
UPDATE_GLOBAL_ITER = 10 # 
GAMMA = 0.99
MAX_EP = 500
hidden_dim_pi = 16
hidden_dim_v = 16
env = gym.make('CartPole-v0')
N_S = env.observation_space.shape[0]
N_A = env.action_space.n

# Define basic neural network(It will be same for each worker)
class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim # 4
        self.a_dim = a_dim # 2
        self.pi1 = nn.Linear(s_dim, hidden_dim_pi) # (N, 4) -> (N, hidden_dim_pi)
        self.pi2 = nn.Linear(hidden_dim_pi, a_dim) # (N, hidden_dim_pi) -> (N, 2)
        self.v1 = nn.Linear(s_dim, hidden_dim_v) # (N, 4) -> (N, hidden_dim_v)
        self.v2 = nn.Linear(hidden_dim_v, 1) # (N, hidden_dim_v) -> (N, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical # It means that [a, b, c, ...] -> 0:a, 1:b, 2:c, ...
    # forward returns output of model
    # Return : softmax^(-1)(probability) and V(s)  (Note. During using crossentropy loss in pytorch, network must not contain softmax layer)
    def forward(self, x):
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values
    # choose_action returns action from state s
    # Return : action 
    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data # We need to change to probability
        m = self.distribution(prob) # take actions by given probability
        return m.sample().numpy()[0]
    # evaluate loss function
    # v_t : r+gamma*v_(t+1)
    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)           # local network
        self.env = gym.make('CartPole-v0').unwrapped

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True: #현재 시점 t
                if self.name == 'w00':
                    self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a)
                if done: r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        ep_r = min(ep_r, 200)
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)


if __name__ == "__main__":
    gnet = Net(N_S, N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=5e-4, betas=(0.92, 0.999))      # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt
    res = np.array(res)
    np.save("discrete_result.npy", res)
    plt.plot(res)
    plt.ylabel('ep reward')
    plt.xlabel('Episode')
    plt.show()
    