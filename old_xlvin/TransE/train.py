import argparse
import torch
import datetime
import os
import sys

import numpy as np

from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import models
import dataset_util

import logging
import pickle

from tqdm import tqdm
# TODO make output with tqdm not hard coded


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=1024,
                    help='Batch size.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs.')
parser.add_argument('--learning-rate', type=float, default=5e-4,
                    help='Learning rate.')

parser.add_argument('--starting_model', type=str, default=None,
                    help='Wights from previous training')

parser.add_argument('--sigma', type=float, default=0.5,
                    help='Energy scale.')
parser.add_argument('--hinge', type=float, default=1.,
                    help='Hinge threshold parameter.')

parser.add_argument('--hidden-dim', type=int, default=128,
                    help='Number of hidden units in transition MLP.')
parser.add_argument('--embedding-dim', type=int, default=10,
                    help='Dimensionality of embedding.')
parser.add_argument('--action-dim', type=int, default=8,
                    help='Dimensionality of action space.')
parser.add_argument('--num_conv', type=int, default=3,
                    help='number of convolutions run with encoder')
parser.add_argument('--encoder_num_linear', type=int, default=2,
                    help='number of linear layers after convolution (not counting final proj layer)')
parser.add_argument('--transe_vin_type', type=str, default='pool', choices=["pool", "cat", "slice"])

parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 42).')
parser.add_argument('--log-interval', type=int, default=75,
                    help='How many batches to wait before logging'
                         'training status.')
parser.add_argument('--dataset', type=str,
                    default='gridworld_8x8.npz',
                    help='Path to dataset.')
parser.add_argument('--name', type=str, default='none',
                    help='Experiment name.')
parser.add_argument('--save-folder', type=str,
                    default='checkpoints',
                    help='Path to checkpoints.')

parser.add_argument('--new_sample', default=False, action='store_true',
                    help='fine tuned sampleing of bad states')


parser.add_argument('--reward_loss', default=False, action='store_true',
                    help='include loss for predicting reward')

parser.add_argument('--encoder_type', type=str, default='original', choices=['original', 'general'],
                    help='Which encoder to use: original - Encoder, general-SizeAgnosticEncoder')
parser.add_argument('--pool_type', type=str, default='mean', choices=['max', 'mean'],
                    help='If using SizeAgnosticEncoder, the global pooling type needs to be specified.')

args = parser.parse_args()

gpu_use = torch.cuda.is_available()

now = datetime.datetime.now()
timestamp = now.isoformat()

if args.name == 'none':
    exp_name = timestamp
else:
    exp_name = args.name

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if gpu_use:
    torch.cuda.manual_seed(args.seed)

exp_counter = 0
save_folder = '{}/{}/'.format(args.save_folder, exp_name)

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
meta_file = os.path.join(save_folder, 'metadata.pkl')
model_file = os.path.join(save_folder, 'model.pt')
log_file = open(os.path.join(save_folder, 'log.txt'), 'a')
tb_writer = SummaryWriter(log_dir=save_folder)

# logging.basicConfig(level=logging.INFO, format='%(message)s')
# logger = logging.getLogger()
# logger.addHandler(logging.FileHandler(log_file, 'a'))
# print = logger.info

pickle.dump({'args': args}, open(meta_file, "wb"))

device = torch.device('cuda' if gpu_use else 'cpu')

print(f'using: {device}')
print(f'using: {device}', file=log_file)

train_data, valid_data  = dataset_util.load_data(
    fname=args.dataset, 
    new_sample=args.new_sample,
    include_reward=args.reward_loss)

train_size = len(train_data)
valid_size = len(valid_data)

train_loader = data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

valid_loader = data.DataLoader(
    valid_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

# Get data sample
obs = train_loader.__iter__().next()[0]
input_shape = obs[0].size()

model = models.TransE(
    embedding_dim=args.embedding_dim,
    hidden_dim=args.hidden_dim,
    action_dim=args.action_dim,
    input_dims=(3, input_shape[1], input_shape[2]),
    sigma=args.sigma,
    hinge=args.hinge,
    num_conv=args.num_conv,
    new_sample=args.new_sample,
    encoder_type=args.encoder_type,
    pool_type=args.pool_type,
    encoder_num_linear=args.encoder_num_linear,
    transe_vin_type=args.transe_vin_type).to(device)
print("model ", model)
print("model ", model, file=log_file)

if args.starting_model != None:
    model.load_state_dict(torch.load(f'./checkpoints/{args.starting_model}/model.pt'))
    print(f'loaded {args.starting_model} weights')
    print(f'loaded {args.starting_model} weights', file=log_file)
    
else:
    model.apply(models.weights_init)
    print('weights initialized')
    print('weights initialized', file=log_file)

parameters = list(model.parameters())
if args.reward_loss:
    reward_predictor = models.RewardPredictor(model, new_sample=args.new_sample).to(device)
    parameters += list(reward_predictor.parameters())
optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

# Train model.
print('Starting model training...')
print('Starting model training...', file=log_file)
print(f'time is {datetime.datetime.now()}')
print(f'time is {datetime.datetime.now()}', file=log_file)
begining_time = datetime.datetime.now()
step = 0
best_loss = 1e9


for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0
    train_batches = 0
    for batch_idx, data_batch in enumerate(train_loader):
        data_batch = [tensor.to(device) for tensor in data_batch]
    
        optimizer.zero_grad()

        if args.reward_loss:
            data_batch, reward = data_batch[:-1], data_batch[-1]
            reward_loss = reward_predictor.reward_loss(data_batch, reward)
            tb_writer.add_scalar('Train/Loss/reward', reward_loss.item(), step)
        loss = model.contrastive_loss(*data_batch)
        tb_writer.add_scalar('Train/Loss/contrastive', loss.item(), step)

        if args.reward_loss:
            with torch.no_grad():
                predicted_reward = reward_predictor(data_batch)
                avg_rew = predicted_reward.mean()
                tb_writer.add_scalar('Train/reward_avg', avg_rew.item(), step)
            loss += reward_loss
        
        tb_writer.add_scalar('Train/Loss/total', loss.item(), step)

        loss.backward()
        if args.reward_loss:
            grad_mean = 0.
            num_pam = 0
            for p in reward_predictor.parameters():
                grad_mean += p.grad.abs().sum()
                num_pam += p.numel()
            grad_mean /= num_pam
            tb_writer.add_scalar('Grads/reward', grad_mean.item(), step)

        grad_mean = 0.
        num_pam = 0
        for p in model.parameters():
            grad_mean += p.grad.abs().sum()
            num_pam += p.numel()
        grad_mean /= num_pam
        tb_writer.add_scalar('Grads/model', grad_mean.item(), step)
        
        train_loss += loss.item()
        optimizer.step()
        train_batches += 1

        if batch_idx % args.log_interval == 0:
            with torch.no_grad():
                valid_loss = 0
                valid_contra = 0
                valid_reward = 0
                number_of_batches = 0
                for data_batch in valid_loader:
                    data_batch = [tensor.to(device) for tensor in data_batch]
                    if args.reward_loss:
                        data_batch, reward = data_batch[:-1], data_batch[-1]
                        reward_loss = reward_predictor.reward_loss(data_batch, reward)
                        valid_reward += reward_loss.item()
                        predicted_reward = reward_predictor(data_batch)
                        avg_rew = predicted_reward.mean()
                        tb_writer.add_scalar('Valid/reward_avg', avg_rew.item(), step)
                    loss = model.contrastive_loss(*data_batch)

                    
                    valid_contra += loss.item()
                    if args.reward_loss:
                        loss += reward_loss
                    valid_loss += loss.item()
                    number_of_batches += 1
                
                
                print('Epoch: {} [{}/{}]\tvalidation loss: {:.6f}'.format(
                        epoch, batch_idx * args.batch_size,
                        len(train_loader.dataset),
                        valid_loss / number_of_batches))
                print('Epoch: {} [{}/{}]\tvalidation loss: {:.6f}'.format(
                        epoch, batch_idx * args.batch_size,
                        len(train_loader.dataset),
                        valid_loss / number_of_batches), file=log_file)
                
                if args.reward_loss:
                    tb_writer.add_scalar('Valid/Loss/reward', valid_reward / number_of_batches, step)
                tb_writer.add_scalar('Valid/Loss/contrastive', valid_contra / number_of_batches, step)
                tb_writer.add_scalar('Valid/Loss/total', valid_loss / number_of_batches, step)

        step += 1


    valid_loss = 0
    valid_contra = 0
    valid_reward = 0
    number_of_batches = 0
    for data_batch in valid_loader:
        with torch.no_grad():
            data_batch = [tensor.to(device) for tensor in data_batch]
            if args.reward_loss:
                data_batch, reward = data_batch[:-1], data_batch[-1]
                reward_loss = reward_predictor.reward_loss(data_batch, reward)
                valid_reward += reward_loss.item()
            loss = model.contrastive_loss(*data_batch)
            valid_contra += loss.item()
            if args.reward_loss:
                loss += reward_loss
            valid_loss += loss.item()
            number_of_batches += 1
    
    print('====> Epoch: {} train loss: {:.6g} validation loss: {:.6f}\n'.format(
        epoch, train_loss / train_batches , valid_loss / number_of_batches))
    print('====> Epoch: {} train loss: {:.6g} validation loss: {:.6f}\n'.format(
        epoch, train_loss / train_batches , valid_loss / number_of_batches), file=log_file)
                
    tb_writer.add_scalar('Valid/Loss/contrastive', valid_contra / number_of_batches, step)
    tb_writer.add_scalar('Valid/Loss/total', valid_loss / number_of_batches, step)
    if args.reward_loss:
        tb_writer.add_scalar('Valid/Loss/reward', valid_reward / number_of_batches, step)


    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(model.state_dict(), model_file)

print('training done')
print('training done', file=log_file)
print(f'time is {datetime.datetime.now()}')
print(f'time is {datetime.datetime.now()}', file=log_file)
print(f'training took {(datetime.datetime.now() - begining_time).total_seconds() / 3600 :.2f} hours')
print(f'training took {(datetime.datetime.now() - begining_time).total_seconds() / 3600 :.2f} hours', file=log_file)
