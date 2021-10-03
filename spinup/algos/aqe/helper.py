import numpy as np
import torch
from torch import Tensor


def get_probabilistic_num_min(num_mins):
    floored_num_mins = np.floor(num_mins)
    if num_mins - floored_num_mins > 0.001:
        prob_for_higher_value = num_mins - floored_num_mins
        if np.random.uniform(0, 1) < prob_for_higher_value:
            return int(floored_num_mins+1)
        else:
            return int(floored_num_mins)
    else:
        return num_mins

def get_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor, q_target_mode, q_target_net_list,
                         policy_net, num_Qs, num_mins, alpha, gamma, q_weights=None,
                         weighted_noise_scale=None, drop_num=None):
    # allow min as a float:
    num_mins_to_use = get_probabilistic_num_min(num_mins)
    sample_idxs = np.random.choice(num_Qs, num_mins_to_use, replace=False)
    with torch.no_grad():
        if q_target_mode == 'min':
            a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = policy_net.forward(obs_next_tensor)
            """sample 2 Qs from the Q list and compute the min Q"""
            q_prediction_next_list = []
            for sample_idx in sample_idxs:
                q_prediction_next = q_target_net_list[sample_idx](torch.cat([obs_next_tensor, a_tilda_next], 1))
                q_prediction_next_list.append(q_prediction_next)
            q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)
            min_q, min_indices = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
            next_q_with_log_prob = min_q - alpha * log_prob_a_tilda_next
            y_q = rews_tensor + gamma * (1 - done_tensor) * next_q_with_log_prob

        if q_target_mode == 'ave':
            a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = policy_net.forward(obs_next_tensor)
            q_prediction_next_list = []
            for q_i in range(num_Qs):
                q_prediction_next = q_target_net_list[q_i](torch.cat([obs_next_tensor, a_tilda_next], 1))
                q_prediction_next_list.append(q_prediction_next)
            q_prediction_next_ave = torch.cat(q_prediction_next_list, 1).mean(dim=1).reshape(-1, 1)
            next_q_with_log_prob = q_prediction_next_ave - alpha * log_prob_a_tilda_next
            y_q = rews_tensor + gamma * (1 - done_tensor) * next_q_with_log_prob

        if q_target_mode == 'rem':
            a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = policy_net.forward(obs_next_tensor)
            q_prediction_next_list = []
            for q_i in range(num_Qs):
                q_prediction_next = q_target_net_list[q_i](torch.cat([obs_next_tensor, a_tilda_next], 1))
                q_prediction_next_list.append(q_prediction_next)
            # apply rem here
            q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)
            rem_weight = Tensor(np.random.uniform(0, 1, q_prediction_next_cat.shape)).to(device=policy_net.last_fc_layer.weight.device)
            normalize_sum = rem_weight.sum(1).reshape(-1, 1).expand(-1, num_Qs)
            rem_weight = rem_weight / normalize_sum
            q_prediction_next_rem = (q_prediction_next_cat * rem_weight).sum(dim=1).reshape(-1, 1)
            next_q_with_log_prob = q_prediction_next_rem - alpha * log_prob_a_tilda_next
            y_q = rews_tensor + gamma * (1 - done_tensor) * next_q_with_log_prob

        if q_target_mode == 'aqe':
            a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = policy_net.forward(obs_next_tensor)
            q_prediction_next_list = []
            for q_i in range(num_Qs):
                q_prediction_next = q_target_net_list[q_i](torch.cat([obs_next_tensor, a_tilda_next], 1))
                q_prediction_next_list.append(q_prediction_next)
            q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)
            #reorder:
            increasing_q_next_vals, _ = torch.sort(q_prediction_next_cat, dim=1)

            #drop outliers:
            if drop_num == 0:
                dropped_q_next_vals = increasing_q_next_vals
            else:
                dropped_q_next_vals = increasing_q_next_vals[:,: - drop_num]

            #average of remaining values
            q_prediction_next_ave = torch.mean(dropped_q_next_vals, 1, True)
            next_q_with_log_prob = q_prediction_next_ave - alpha * log_prob_a_tilda_next
            y_q = rews_tensor + gamma * (1 - done_tensor) * next_q_with_log_prob

        if q_target_mode == 'median':
            a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = policy_net.forward(obs_next_tensor)
            q_prediction_next_list = []
            for q_i in range(num_Qs):
                q_prediction_next = q_target_net_list[q_i](torch.cat([obs_next_tensor, a_tilda_next], 1))
                q_prediction_next_list.append(q_prediction_next)
            q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)
            # reorder:
            #increasing_q_next_vals, _ = torch.sort(q_prediction_next_cat, dim=1)
            q_prediction_next_median, _ = torch.median(q_prediction_next_cat, dim=1, keepdim=True)
            next_q_with_log_prob = q_prediction_next_median - alpha * log_prob_a_tilda_next
            y_q = rews_tensor + gamma * (1 - done_tensor) * next_q_with_log_prob

        if q_target_mode == 'outlier': #drop the minimum and the maximum
            a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = policy_net.forward(obs_next_tensor)
            q_prediction_next_list = []
            for q_i in range(num_Qs):
                q_prediction_next = q_target_net_list[q_i](torch.cat([obs_next_tensor, a_tilda_next], 1))
                q_prediction_next_list.append(q_prediction_next)
            q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)
            # reorder:
            increasing_q_next_vals, _ = torch.sort(q_prediction_next_cat, dim=1)
            dropped_q_next_vals = increasing_q_next_vals[:, 1: -1]

            q_prediction_next_ave = torch.mean(dropped_q_next_vals, 1, True)
            next_q_with_log_prob = q_prediction_next_ave - alpha * log_prob_a_tilda_next
            y_q = rews_tensor + gamma * (1 - done_tensor) * next_q_with_log_prob

        if q_target_mode == 'minpair':
            n_pair = int(num_Qs/2)
            first_idx = np.random.choice(n_pair, 1, replace=False)[0]
            second_idx = first_idx + n_pair
            sample_idxs = [first_idx, second_idx]
            """following is same as min mode"""
            a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = policy_net.forward(obs_next_tensor)
            """sample 2 Qs from the Q list and compute the min Q"""
            q_prediction_next_list = []
            for sample_idx in sample_idxs:
                q_prediction_next = q_target_net_list[sample_idx](torch.cat([obs_next_tensor, a_tilda_next], 1))
                q_prediction_next_list.append(q_prediction_next)
            q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)
            min_q, min_indices = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
            next_q_with_log_prob = min_q - alpha * log_prob_a_tilda_next
            y_q = rews_tensor + gamma * (1 - done_tensor) * next_q_with_log_prob

    return y_q, sample_idxs

def get_mc_return_with_entropy_on_reset(mc_env, deterministic, policy_net, max_ep_len, alpha,
                                        gamma, n_total_evalute, mc_cut_off=300):
    # since we want to also compute bias, so we need to
    final_mc_list = np.zeros(0)
    final_mc_entropy_list = np.zeros(0)
    final_obs_list = []
    final_act_list = []
    while final_mc_list.shape[0] < n_total_evalute:
        # we continue if haven't collected enough data
        o = mc_env.reset()
        # temporary lists
        reward_list, log_prob_a_tilda_list, obs_list, act_list = [], [], [], []
        r, d, ep_ret, ep_len = 0, False, 0, 0
        discounted_return = 0
        discounted_return_with_entropy = 0
        for i_step in range(max_ep_len): # run an episode
            with torch.no_grad():
                o_tensor = Tensor(o).unsqueeze(0).to(policy_net.last_fc_layer.weight.device)
                a_tilda_tensor, _, _, log_prob_a_tilda, _, _, = policy_net.forward(o_tensor,deterministic=deterministic)
                a = a_tilda_tensor.cpu().numpy().reshape(-1)
            obs_list.append(o)
            act_list.append(a)
            o, r, d, _ = mc_env.step(a)
            ep_ret += r
            ep_len += 1
            reward_list.append(r)
            log_prob_a_tilda_list.append(log_prob_a_tilda.item())
            if d or (ep_len == max_ep_len):
                break
        dis_ret_list = np.zeros(ep_len)
        dis_ret_ent_list = np.zeros(ep_len)
        for i_step in range(ep_len-1, -1, -1):
            # backwards compute discounted return and with entropy for all s-a visited
            if i_step == ep_len-1:
                dis_ret_list[i_step] = reward_list[i_step]
                dis_ret_ent_list[i_step] = reward_list[i_step]
            else:
                dis_ret_list[i_step] = reward_list[i_step] + gamma * dis_ret_list[i_step + 1]
                dis_ret_ent_list[i_step] = reward_list[i_step] + \
                    gamma * (dis_ret_ent_list[i_step + 1] - alpha * log_prob_a_tilda_list[i_step+1])
        # now we take the first few of these.
        final_mc_list = np.concatenate((final_mc_list, dis_ret_list[:mc_cut_off]))
        final_mc_entropy_list = np.concatenate((final_mc_entropy_list, dis_ret_ent_list[:mc_cut_off]))
        final_obs_list += obs_list[:mc_cut_off]
        final_act_list += act_list[:mc_cut_off]
    return final_mc_list, final_mc_entropy_list, np.array(final_obs_list), np.array(final_act_list)

def soft_update_model1_with_model2(model1, model2, rou):
    """
    see openai spinup sac psudocode line 16, used to update target_value_net
    :param model1: a pytorch model
    :param model2: a pytorch model of the same class
    :param rou: the update is model1 <- rou*model1 + (1-rou)model2
    """
    for model1_param, model2_param in zip(model1.parameters(), model2.parameters()):
        model1_param.data.copy_(
            rou*model1_param.data + (1-rou)*model2_param.data
        )