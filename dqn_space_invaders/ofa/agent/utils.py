import os

import torch


def init_net(net, target_net, save_path='params.pkl'):
    if os.path.exists(save_path):
        print('Parameters exists.\nInit agent from exists parameters...')
        # initialize our agent using the exist model if we have one.
        net.load_state_dict(torch.load(save_path))
    else:
        print('Create new parameters...')
        torch.save(net.state_dict(), save_path)

    if target_net is not None:
        update_target(target_net, save_path)


def save_net(net, save_path='params.pkl'):
    torch.save(net.state_dict(), save_path)


def update_target(target_net, save_path='params.pkl'):
    target_net.load_state_dict(torch.load(save_path))


def set_device():
    if torch.cuda.is_available():
        print('CUDA backend enabled.')
        device = torch.device('cuda')
    else:
        print('CPU backend enabled.')
        device = torch.device('cpu')
    return device
