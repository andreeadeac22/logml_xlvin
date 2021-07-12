from tqdm import tqdm
import numpy as np
import torch
import h5py
import os
import random


def _load_existing_dataset(file_name, train_portion, new_sample,
                         include_reward=False):
    with np.load(file_name, mmap_mode='r') as f:
        images = f['arr_0']
    
    nb_images = len(images)
    
    train_images = images[:int(nb_images * train_portion)]
    valid_images = images[int(nb_images * train_portion):]  

    train_exp_buff = []

    valid_exp_buff = []
    

    print('loading training set:')
    nums = np.array([0, 0, 0])
    
    for img in tqdm(train_images):
        tr, x =  _get_transitions(img, new_sample, include_reward)
        nums += x
        train_exp_buff += tr
        # if len(train_exp_buff) > 10:
        #     break

    print(nums)


    print('loading validation set:')
    nums = np.array([0, 0, 0])
    print('validation set:')
    for img in tqdm(valid_images):
        tr, x =  _get_transitions(img, new_sample, include_reward)
        nums += x
        valid_exp_buff += tr
        # if len(valid_exp_buff) > 30:
        #     break
    
    print(nums)

    
    print(f'Number of Train Samples: {len(train_exp_buff)}')
    print(f'Number of Valid Samples: {len(valid_exp_buff)}')


    return train_exp_buff, valid_exp_buff


def _create_train_dataset(file_name='gridworld_8x8.npz', train_portion=0.8, 
                          new_sample=False, include_reward=False):
    
    with np.load(file_name, mmap_mode='r') as f:
        images = f['arr_0']
        
    images = images.astype(np.float32)
    valid_images = []
    for img in tqdm(images):
        for vimg in valid_images:
            if np.array_equal(img[0], vimg[0]) \
            and np.array_equal(img[1], vimg[1]):
                    break
        else:
            valid_images.append(img)
    
    images = valid_images

    np.savez(f'cleanup_{file_name}', images)

    nb_images = len(images)
    nb_actions = 8
    nb_states = 64

    train_images = images[:int(nb_images * train_portion)]
    valid_images = images[int(nb_images * train_portion):]    

    print('training set:')
    train_exp_buff = []

    valid_exp_buff = []
    nums = np.array([0, 0, 0])
    
    for img in tqdm(train_images):
        tr, x =  _get_transitions(img, new_sample, include_reward)
        nums += x
        train_exp_buff += tr
        if len(train_exp_buff) > 1000000:
            break

    print(nums)

    nums = np.array([0, 0, 0])
    print('validation set:')
    for img in tqdm(valid_images):
        tr, x =  _get_transitions(img, new_sample, include_reward)
        nums += x
        valid_exp_buff += tr
        if len(valid_exp_buff) > 300000:
            break
    
    print(nums)
    
    print(f'Number of Train Samples: {len(train_exp_buff)}')
    print(f'Number of Valid Samples: {len(valid_exp_buff)}')


    return train_exp_buff, valid_exp_buff

def _get_transitions(img, new_sample, include_reward):
    if new_sample:
        return _get_transitions_new(img, include_reward)
    else:
        return _get_transitions_old(img, include_reward)

def _get_transitions_old(img, include_reward):
    exp_buff = []


    dxs = [1, 0, -1, 0, 1, 1, -1, -1]
    dys = [0, 1, 0, -1, 1, -1, 1, -1]

    grid = img[0]
    reward_map = img[1]
    
    valid_states = set()
    
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 0.0:
                valid_states.add((x, y))
        
    nums = np.array([0, 0, 0])

    for x, y in valid_states:
        current_position = np.zeros(shape=(grid.shape), dtype=np.float32)
        current_position[x, y] = 1.
        for A in range(len(dxs)):
            dx = dxs[A]
            dy = dys[A]

            next_x = x + dx
            next_y = y + dy
            if (next_x, next_y) not in valid_states:
                next_x = x
                next_y = y
            
            S = np.array([grid, current_position, reward_map])
            
            next_position = np.zeros(shape=(grid.shape), dtype=np.float32)
            next_position[next_x, next_y] = 1.
            S_P = np.array([grid, next_position, reward_map])
            
            if include_reward:
                if (x + dx, y + dy) not in valid_states:
                    nums[0] += 1
                    reward = -1.
                elif reward_map[next_x, next_y] == 10.:
                    nums[1] += 1
                    reward = 1.
                else:
                    nums[2] += 1
                    reward = -0.01
                exp_buff.append((S, A, S_P, reward))
            else:
                exp_buff.append((S, A, S_P))
    return exp_buff, nums

def _get_transitions_new(img, include_reward):
    exp_buff = []


    dxs = [1, 0, -1, 0, 1, 1, -1, -1]
    dys = [0, 1, 0, -1, 1, -1, 1, -1]

    grid = img[0]
    reward_map = img[1]
    
    valid_states = set()
    
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 0.0:
                valid_states.add((x, y))
    nums = np.array([0, 0, 0])

    for x, y in valid_states:
        current_position = np.zeros(shape=(grid.shape), dtype=np.float32)
        current_position[x, y] = 1.
        for A in range(len(dxs)):
            dx = dxs[A]
            dy = dys[A]

            next_x = x + dx
            next_y = y + dy
            if (next_x, next_y) not in valid_states:
                next_x = x
                next_y = y         
            

            fake_x, fake_y = next_x, next_y
            iter_count = 0
            while fake_x == next_x and fake_y == next_y:
                fake_dx = random.randint(-1, 1)
                fake_dy = random.randint(-1, 1)
                fake_x = x + fake_dx
                fake_y = y + fake_dy
                if (fake_x, fake_y) not in valid_states:
                    fake_x = next_x
                    fake_y = next_y
                iter_count += 1
                if iter_count > 15:
                    break
            
            if fake_x == next_x and fake_y == next_y:
                continue
            

            next_position = np.zeros(shape=(grid.shape), dtype=np.float32)
            next_position[next_x, next_y] = 1.

            fake_position = np.zeros(shape=(grid.shape), dtype=np.float32)
            fake_position[fake_x, fake_y] = 1.
            # ['grid', 'start', 'action', 'end', 'fake', 'reward_map']

            if include_reward:
                if (x + dx, y + dy) not in valid_states:
                    reward = -1.
                    nums[0] += 1
                elif reward_map[next_x, next_y] == 10.:
                    reward = 1.
                    nums[1] += 1
                else:
                    reward = -0.01
                    nums[2] += 1
                step = (
                    torch.tensor([grid]),
                    torch.tensor([current_position]),
                    torch.tensor(A),
                    torch.tensor([next_position]),
                    torch.tensor([fake_position]),
                    torch.tensor([reward_map]),
                    torch.tensor(reward),
                )
            else:
                step = (
                    torch.tensor([grid]),
                    torch.tensor([current_position]),
                    torch.tensor(A),
                    torch.tensor([next_position]),
                    torch.tensor([fake_position]),
                    torch.tensor([reward_map]),
                )
            exp_buff.append(step)
    return exp_buff, nums


class DataWrapper(torch.utils.data.Dataset):
    def __init__(self, buffer):
        self.expirience_buffer = buffer
        
    def __len__(self):
        return len(self.expirience_buffer)
    
    def __getitem__(self, idx):
        return self.expirience_buffer[idx]

def load_data(fname='gridworld_8x8.npz', train=True, train_portion=0.8,
              new_sample=False, include_reward=False):
    if os.path.exists(f'cleanup_{fname}'):
        train, validation = _load_existing_dataset(
                                f'cleanup_{fname}', 
                                train_portion=train_portion,
                                new_sample=new_sample,
                                include_reward=include_reward)
    else:
        train, validation = _create_train_dataset(
                                file_name=fname, 
                                train_portion=train_portion,
                                new_sample=new_sample,
                                include_reward=include_reward)
    return DataWrapper(train), DataWrapper(validation)
    