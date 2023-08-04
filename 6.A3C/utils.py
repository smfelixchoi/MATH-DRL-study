"""
Functions that use multiple times
"""

from torch import nn
import torch
import numpy as np

# v_wrap : numpy array를 입력받아 data type을 지정된 형태로 변경하고 이를 torch tensor로 변환함
def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)

# set_init : 층 내부 가중치/편차를 초기화. 가중치는 정규분포 N(0, 0.01), 편차는 0으로 설정
def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)

# push_and_pull : update global network and local network
# opt:optimizer
# lnet: local network
# gnet: global network
# done: Completion at this step
# s_: state at this step
# bs: state batch
# ba: action batch
# br: reward batch
# gamma : discount rate

def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r to calculate previous v's with gamma
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad # Apply local network gradient to global network (_grad means writable)
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())

# Save and print result
def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        global_ep_r.value = ep_r
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.2f" % global_ep_r.value,
    )