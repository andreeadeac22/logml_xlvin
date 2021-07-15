import torch
import torch.nn as nn

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


# TODO: rewrite! It's an old one from old_xlvin/rl/ppo.py

class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 optimizer,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 transe_loss_coef=0.,
                 mini_batch_size=None,
                 transe_detach=False):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.mini_batch_size = mini_batch_size

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.transe_loss_coef = transe_loss_coef

        self.transe_detach = transe_detach

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optimizer  # torch.optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        transe_loss_epoch = 0

        for e in range(self.ppo_epoch):
            # if self.actor_critic.is_recurrent:
            #    data_generator = rollouts.recurrent_generator(
            #        advantages, self.num_mini_batch)
            # else:
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch, self.mini_batch_size)
            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, \
                old_action_log_probs_batch, solved_obs_batch, adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
                    obs_batch, actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                                         (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                if self.transe_loss_coef > 0.:
                    transe_loss = self.transe_loss_coef * \
                                  self.actor_critic.transe.transition_loss(obs_batch, actions_batch.squeeze(-1),
                                                                           solved_obs_batch,
                                                                           self.transe_detach)
                    loss += transe_loss
                    transe_loss_epoch += transe_loss.item()

                loss.backward(retain_graph=True)
                """
                total_norm = 0
                for p in self.actor_critic.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                print("actor_critic ", total_norm)
                """
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        transe_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, transe_loss_epoch


def gather_rollout(env, policy, num_steps, gamma, num_processes, device, deterministic=False, have_solved_state=True):
    rollouts = RolloutStorage(num_steps, env.observation_space.shape, env.action_space, device, num_processes,
                              have_solved_state)
    num_passed = 0
    num_failed = 0
    obs = env.reset()
    rollouts.obs[0].copy_(obs)

    for step in range(num_steps):
        with torch.no_grad():
            value, action, log_probs = policy.act(rollouts.obs[step].to(device), deterministic=deterministic)
        # env.render()
        obs, reward, done, solved_obs = env.step(action)
        reward = torch.tensor(reward)
        mask = ~ torch.tensor(done).unsqueeze_(-1)
        if have_solved_state:
            solved_obs = torch.tensor(solved_obs)
            next_obs = mask.unsqueeze(-1).unsqueeze(-1) * obs + \
                       (~mask.unsqueeze(-1).unsqueeze(-1)) * solved_obs
        else:
            next_obs = None

        num_passed += torch.sum(torch.eq(reward, torch.ones_like(reward)))
        num_failed += torch.sum(torch.eq(reward, torch.ones_like(reward) * -1))
        rollouts.insert(obs, action.squeeze_(0), log_probs.squeeze_(0), value.squeeze_(0), reward, mask, next_obs)
    with torch.no_grad():
        next_value = policy.get_value(rollouts.obs[-1].to(device)).detach()

    use_gae = False
    gae_lambda = 0.01
    rollouts.compute_returns(next_value, use_gae, gamma, gae_lambda)
    return rollouts, num_passed, num_failed


def gather_fixed_ep_rollout(env, policy, num_episodes, gamma, num_processes, device, deterministic=False,
                            have_solved_state=False):
    all_obs = []
    all_actions = []
    all_log_probs = []
    all_values = []
    all_rewards = []
    all_masks = []
    all_next_obs = []

    obs = env.reset()
    all_obs += [obs]
    done_ep = 0
    step = 0
    while done_ep < num_episodes:
        with torch.no_grad():
            value, action, log_probs = policy.act(torch.Tensor(all_obs[-1]).to(device), deterministic=deterministic)
        # env.render()
        obs, reward, done, _ = env.step(action)
        done_ep += torch.sum(torch.tensor(done))

        reward = torch.tensor(reward)
        mask = ~ torch.tensor(done).unsqueeze_(-1)
        next_obs = obs

        all_obs += [obs]
        all_actions += [action.to('cpu')]
        all_log_probs += [log_probs.to('cpu')]
        all_values += [value.to('cpu')]
        all_rewards += [reward]
        all_masks += [mask]

        all_next_obs += [next_obs]

        step += 1

    print("Number of steps ", step)

    rollouts = RolloutStorage(step, env.observation_space.shape, env.action_space, device, num_processes,
                              have_solved_state=True)

    rollouts.obs = torch.stack(all_obs, dim=0)
    rollouts.actions = torch.stack(all_actions, dim=0)
    rollouts.action_log_probs = torch.stack(all_log_probs, dim=0)
    rollouts.value_preds = torch.cat((torch.zeros_like(all_values[0]).unsqueeze(0), torch.stack(all_values, dim=0)),
                                     dim=0)
    rollouts.rewards = torch.stack(all_rewards, dim=0)
    rollouts.masks = torch.cat((torch.zeros_like(all_masks[0]).unsqueeze(0), torch.stack(all_masks, dim=0)), dim=0)

    rollouts.solved_obs = torch.stack(all_next_obs, dim=0)

    with torch.no_grad():
        next_value = policy.get_value(rollouts.obs[-1].to(rollouts.device)).detach()
    use_gae = False
    gae_lambda = 0.01
    rollouts.compute_returns(next_value, use_gae, gamma, gae_lambda)
    return rollouts


class RolloutStorage(object):
    def __init__(self, num_steps, obs_shape, action_space, device, num_processes=1, have_solved_state=True):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)

        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            # action_shape = action_space.shape[0]
            raise NotImplementedError
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.have_solved_state = have_solved_state
        # solved obs, for when env is reset and the new maze is in obs
        if self.have_solved_state:
            self.solved_obs = torch.zeros(num_steps, num_processes, *obs_shape)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        # self.bad_masks = torch.ones(num_steps + 1, 1)

        self.num_steps = num_steps
        self.step = 0
        self.device = device

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks, solved_obs=None):
        self.obs[self.step + 1].copy_(obs)
        self.rewards[self.step].copy_(rewards)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.masks[self.step + 1].copy_(masks)
        if self.have_solved_state:
            self.solved_obs[self.step].copy_(solved_obs)

        self.step = (self.step + 1) % self.num_steps

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[
                    step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                                     gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            obs_batch = obs_batch.to(self.device)
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            actions_batch = actions_batch.to(self.device)
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            value_preds_batch = value_preds_batch.to(self.device)
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            return_batch = return_batch.to(self.device)
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            masks_batch = masks_batch.to(self.device)
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            old_action_log_probs_batch = old_action_log_probs_batch.to(self.device)
            if self.have_solved_state:
                solved_obs_batch = self.solved_obs.view(-1, *self.solved_obs.size()[2:])[indices]
                solved_obs_batch = solved_obs_batch.to(self.device)
            else:
                solved_obs_batch = None

            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]
                adv_targ = adv_targ.to(self.device)

            yield obs_batch, actions_batch, \
                  value_preds_batch, return_batch, \
                  masks_batch, old_action_log_probs_batch, solved_obs_batch, adv_targ
