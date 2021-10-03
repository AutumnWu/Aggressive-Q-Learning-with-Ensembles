import gym
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import time
import sys
from spinup.algos.aqe.core_aqe import TanhGaussianPolicySACAdapt, Mlp, ReplayBuffer
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.algos.aqe.helper import get_probabilistic_num_min, get_q_target_no_grad, \
get_mc_return_with_entropy_on_reset, soft_update_model1_with_model2

"""
AQE: num_Qs = 10, q_target_mode = 'aqe'
SAC: num_Qs = 2, num_mins = 2, q_target_mode = 'min'
REDQ: num_Qs > 2, num_mins = 2, q_target_mode = 'min'
MaxMin: num_mins = num_Qs, q_target_mode = 'min'

Ens Ave: set q_target_mode to 'ave'
REM: set q_target_mode to 'rem'

policy_update_mode: 'min' or 'ave'
"""
def aqe(env_fn, hidden_sizes=[256, 256], seed=0,
         epochs=1000, replay_size=int(1e6), gamma=0.99,
         polyak=0.995, lr=3e-4, alpha=0.2, batch_size=256, start_steps=5000,
         max_ep_len=1000, save_freq=1000, save_model=True,
         auto_alpha=True, target_entropy='auto', grad_clip=-1, logger_store_freq=50,
         steps_per_epoch=1000, n_evals_per_epoch=10,
         update_freq=20, delay_update_steps='auto', delay_update_g=0,
         num_Qs=10, num_mins=2, 
         policy_update_delay=20, polyak_delay=1, q_target_mode='aqe',
         do_analysis = True, do_new_analysis_scheme=True, q_update_delay=1,
         update_prob_each_q=1, n_mc_eval=1000, n_mc_cutoff=350,
         dense_policy=False, mc_deterministic=False, mc_use_start_action=True, mc_start_from_reset=False,
         do_variance_analysis=False, policy_update_mode='ave', data_dir_analysis=None, lamda=0, hidden_size_setting=None,
         q_weight_decay=0, entropy_transition_steps=0, initial_alpha=0.3,
         use_ave_q_in_analysis=True,
         logger_kwargs=dict(), analysis_logger_kwargs=dict(), drop_num=2, multihead=2):
    """
    Largely following OpenAI documentation
    But slightly different from tensorflow implementation
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        hidden_sizes: number of entries is number of hidden layers
            each entry in this list indicate the size of that hidden layer.
            applies to all networks

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch. Note the epoch here is just logging epoch
            so every this many steps a logging to stdouot and also output file will happen
            note: not to be confused with training epoch which is a term used often in literature for all kinds of
            different things

        epochs (int): Number of epochs to run and train agent. Usage of this term can be different in different
            algorithms, use caution. Here every epoch you get new logs

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration. However during testing the action always come from policy

        max_ep_len (int): Maximum length of trajectory / episode / rollout. Environment will get reseted if
        timestep in an episode excedding this number

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        logger_kwargs (dict): Keyword args for EpochLogger.

        dont_save (bool): TODO currently don't support save

        grad_clip: whether to use gradient clipping. < 0 means no clipping

        logger_store_freq: how many steps to log debugging info, typically don't need to change

    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("running on device:", device)
    if num_mins == 'same':
        num_mins = num_Qs
    if delay_update_steps == 'auto':
        delay_update_steps = start_steps

    hidden_size_setting_list = [[256, 256], [512, 512, 512]]
    if hidden_size_setting is not None:
        hidden_sizes = hidden_size_setting_list[hidden_size_setting]

    """set up logger"""
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    exp_name_analysis = logger_kwargs["exp_name"] + "_analysis"
    analysis_logger_kwargs = setup_logger_kwargs(exp_name_analysis, seed, data_dir=data_dir_analysis)
    analysis_logger = EpochLogger(**analysis_logger_kwargs)
    analysis_logger.save_config(locals())

    env, test_env, mc_return_env, val_env = env_fn(), env_fn(), env_fn(), env_fn()
    env_name = env.unwrapped.spec.id

    ## seed torch and numpy
    torch.manual_seed(seed)
    np.random.seed(seed)

    ## seed environment along with env action space so that everything about env is seeded
    env.seed(seed)
    env.action_space.np_random.seed(seed)
    test_env.seed(seed)
    test_env.action_space.np_random.seed(seed)
    mc_return_env.seed(seed)
    mc_return_env.action_space.np_random.seed(seed)
    val_env.seed(seed)
    val_env.action_space.np_random.seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # if environment has a smaller max episode length, then use the environment's max episode length
    max_ep_len = env._max_episode_steps if max_ep_len > env._max_episode_steps else max_ep_len

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    # we need .item() to convert it from numpy float to python float
    act_limit = env.action_space.high[0].item()

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    replay_buffer_val = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    reset_state_list = [] if do_analysis else None # use this to store states for reset

    """
    Auto tuning alpha
    """
    if auto_alpha:
        if target_entropy == 'auto':
            target_entropy = -np.prod(env.action_space.shape).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = optim.Adam([log_alpha], lr=lr)
        alpha = log_alpha.cpu().exp().item()
    else:
        target_entropy, log_alpha, alpha_optim = None, None, None
    print('target entropy:', target_entropy)

    def test_agent(n):
        """
        This will test the agent's performance by running n episodes
        During the runs, the agent only take deterministic action, so the
        actions are not drawn from a distribution, but just use the mean
        :param n: number of episodes to run the agent
        """
        ep_return_list = np.zeros(n)
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                a = policy_net.get_env_action(o, deterministic=True)
                o, r, d, _ = test_env.step(a)
                ep_ret += r
                ep_len += 1
            ep_return_list[j] = ep_ret
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    """init all networks"""

    # set up networks
    
    policy_net = TanhGaussianPolicySACAdapt(obs_dim, act_dim, hidden_sizes, action_limit=act_limit).to(device)
    
    q_net_list, q_target_net_list = [], []
    for q_i in range(num_Qs):
        if multihead > 1:
            new_q_net = Mlp(obs_dim + act_dim, multihead, hidden_sizes).to(device)
            new_q_target_net = Mlp(obs_dim + act_dim, multihead, hidden_sizes).to(device)
        else:
            new_q_net = Mlp(obs_dim + act_dim, 1, hidden_sizes).to(device)
            new_q_target_net = Mlp(obs_dim + act_dim, 1, hidden_sizes).to(device)

        q_net_list.append(new_q_net)
        new_q_target_net.load_state_dict(new_q_net.state_dict())
        q_target_net_list.append(new_q_target_net)

    # set up optimizers
    policy_optimizer = optim.Adam(policy_net.parameters(),lr=lr)
    q_optimizer_list = []
    for q_i in range(num_Qs):
        q_optimizer_list.append(optim.Adam(q_net_list[q_i].parameters(),lr=lr, weight_decay=q_weight_decay))

    # mean squared error loss for v and q networks
    mse_criterion = nn.MSELoss()

    # set up masking
    mask_tensor = torch.zeros((replay_size, num_Qs), device=device)
    uniform_rand = torch.rand((replay_size,num_Qs))
    mask_tensor[uniform_rand<update_prob_each_q] = 1 # 1 at i,j indicates i th data will be used by network j

    # Main loop: collect experience in env and update/log each epoch
    # NOTE: t here is the current number of total timesteps used
    # it is not the number of timesteps passed in the current episode
    current_update_index = 0

    for t in range(total_steps):
        # save state for reset
        
        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        if t > start_steps:
            a = policy_net.get_env_action(o, deterministic=False)
        else:
            a = env.action_space.sample()

        # Step the env, get next observation, reward and done signal
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d


        # Store experience (observation, action, reward, next observation, done) to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2
        """
        Main loop
        """
        # Do G gradient updates per env interact
        G = delay_update_g if t <= delay_update_steps else update_freq
        for j in range(G):
            # get data from replay buffer
            batch = replay_buffer.sample_batch(batch_size)
            obs_tensor = Tensor(batch['obs1']).to(device)
            obs_next_tensor = Tensor(batch['obs2']).to(device)
            acts_tensor = Tensor(batch['acts']).to(device)
            # unsqueeze is to make sure rewards and done tensors are of the shape nx1, instead of n
            # to prevent problems later
            rews_tensor = Tensor(batch['rews']).unsqueeze(1).to(device)
            done_tensor = Tensor(batch['done']).unsqueeze(1).to(device)
            idxs = batch['idxs']

            """
            now we do a SAC update, following the OpenAI spinup doc
            check the openai sac document psudocode part for reference
            line nubmers indicate lines in psudocode part
            we will first compute each of the losses
            and then update all the networks in the end
            """
            # see line 12: get a_tilda, which is newly sampled action (not action from replay buffer)
            """get q loss"""
            # TODO noise injection to weights
            y_q, sample_idxs = get_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor, q_target_mode, q_target_net_list,
                                                    policy_net, num_Qs, num_mins, alpha, gamma, drop_num=drop_num)
            q_loss_val_list = []
            q_prediction_list = []
            for q_i in range(num_Qs):
                q_prediction = q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
                q_prediction_list.append(q_prediction)
            q_prediction_cat = torch.cat(q_prediction_list, dim=1)
            #y_q = y_q.expand((-1,num_Qs)) if y_q.shape[1] == 1 else y_q
            y_q = y_q.expand((-1, num_Qs * max(1, multihead))) if y_q.shape[1] == 1 else y_q
            if update_prob_each_q < 1:
                q_loss_all = mse_criterion(q_prediction_cat * mask_tensor[idxs], y_q * mask_tensor[idxs]) * num_Qs
            else:
                q_loss_all = mse_criterion(q_prediction_cat, y_q) * num_Qs

            """
            get policy loss
            """
            if ((j+1) % policy_update_delay == 0) or j==G-1:
                a_tilda, mean_a_tilda, log_std_a_tilda, log_prob_a_tilda, std, pretanh = policy_net.forward(obs_tensor)

                if policy_update_mode == 'min':
                    q_a_tilda_list = []
                    for sample_idx in sample_idxs:
                        q_a_tilda = q_net_list[sample_idx](torch.cat([obs_tensor, a_tilda], 1))
                        q_a_tilda_list.append(q_a_tilda)
                    q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
                    min_q, min_indices = torch.min(q_a_tilda_cat, dim=1, keepdim=True)
                    policy_loss = (alpha * log_prob_a_tilda - min_q).mean()
                if policy_update_mode == 'ave':
                    q_a_tilda_list = []
                    for sample_idx in range(num_Qs):
                        q_a_tilda = q_net_list[sample_idx](torch.cat([obs_tensor, a_tilda], 1))
                        q_a_tilda_list.append(q_a_tilda)
                    q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
                    ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
                    policy_loss = (alpha * log_prob_a_tilda - ave_q).mean()


                """
                alpha loss, update alpha
                """
                if auto_alpha:
                    if entropy_transition_steps > 0 and t < entropy_transition_steps:
                        # interpolate between 0 and the given target entropy
                        target_entropy_ratio = 1 - (entropy_transition_steps - t) / entropy_transition_steps
                    else:
                        target_entropy_ratio = 1
                    alpha_loss = -(log_alpha * (log_prob_a_tilda + target_entropy).detach()).mean()

                    alpha_optim.zero_grad()
                    alpha_loss.backward()
                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(log_alpha, grad_clip)
                    alpha_optim.step()
                    alpha_auto = log_alpha.cpu().exp().item()
                    alpha = target_entropy_ratio * alpha_auto + (1-target_entropy_ratio) * initial_alpha
                else:
                    alpha_loss = Tensor([0])

            """update networks"""
            if ((j + 1) % q_update_delay == 0) or j == G - 1:
                for q_i in range(num_Qs):
                    q_optimizer_list[q_i].zero_grad()
                q_loss_all.backward()
                for q_i in range(num_Qs):
                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(q_net_list[q_i].parameters(), grad_clip)
                    q_optimizer_list[q_i].step()

            if ((j+1) % policy_update_delay == 0) or j==G-1:
                policy_optimizer.zero_grad()
                policy_loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(policy_net.parameters(), grad_clip)
                policy_optimizer.step()

            # see line 16: update target value network with value network
            if ((j+1) % polyak_delay == 0) or j==G-1:
                for q_i in range(num_Qs):
                    soft_update_model1_with_model2(q_target_net_list[q_i], q_net_list[q_i], polyak)

            # if t % 10==0 and j==5:
            #     print('bp all', bp_time_all,', bp sep', bp_time_sep, ', pi:',policy_time, 'polyak time:',polyak_time)
            current_update_index += 1

            if j==G-1 and t % logger_store_freq == 0:
                # store diagnostic info to logger, IMPORTANT: use reshape(-1) to prevent logger problems
                logger.store(LossPi=policy_loss.cpu().item(), LossQ1=q_loss_all.cpu().item()/num_Qs,
                             LossAlpha=alpha_loss.cpu().item(),
                             Q1Vals=q_prediction.cpu().detach().numpy(),
                             Std=std.cpu().detach().numpy().reshape(-1),
                             Alpha=alpha,
                             LogPi=log_prob_a_tilda.cpu().detach().numpy(),
                             PreTanh=pretanh.abs().cpu().detach().numpy().reshape(-1)
                             )

        if d or (ep_len == max_ep_len):
            ## store episode return and length to logger
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            ## reset environment
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0


        # End of epoch wrap-up
        if (t+1) % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            """
            Save pytorch model, very different from tensorflow version
            We need to save the environment, the state_dict of each network
            and also the state_dict of each optimizer
            """
            # TODO will fix it later saving has not been fully tested
            if save_model:
                #pass
                sac_state_dict = {'env':env, 'policy_net':policy_net.state_dict()}
                             
                if (epoch % save_freq == 0) or (epoch == epochs-1):
                     logger.save_state(sac_state_dict, epoch)
            if current_update_index <= 0: # if haven't done any updates, then log dummy values to prevent problems
                logger.store(Q1Vals=0, Std=0, Alpha=0, LossAlpha=0,
                             LogPi=0, LossPi=0, LossQ1=0, PreTanh=0)

            # Test the performance of the deterministic version of the agent.
            test_agent(n=n_evals_per_epoch)

            # new analysis scheme
            if do_analysis and do_new_analysis_scheme:
                final_mc_list, final_mc_entropy_list, final_obs_list, final_act_list = get_mc_return_with_entropy_on_reset(
                    mc_return_env, mc_deterministic, policy_net, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff)
                logger.store(MCDisRet = final_mc_list)
                logger.store(MCDisRetEnt = final_mc_entropy_list)
                obs_tensor = Tensor(final_obs_list).to(device)
                acts_tensor = Tensor(final_act_list).to(device)
                with torch.no_grad():
                    if multihead > 1:
                        q_prediction = torch.mean(q_net_list[0](torch.cat([obs_tensor, acts_tensor], 1)),
                                                  dim=1).cpu().numpy().reshape(-1)
                    else:
                        q_prediction = q_net_list[0](torch.cat([obs_tensor, acts_tensor], 1)).cpu().numpy().reshape(-1)
                    #q_prediction = q_net_list[1](torch.cat([obs_tensor, acts_tensor], 1)).cpu().numpy().reshape(-1)
                bias = q_prediction - final_mc_entropy_list
                bias_abs = np.abs(bias)
                bias_squared = bias ** 2
                logger.store(QPred=q_prediction)
                logger.store(QBias=bias)
                logger.store(QBiasAbs=bias_abs)
                logger.store(QBiasSqr=bias_squared)
                final_mc_entropy_list_normalize_base = final_mc_entropy_list.copy()
                final_mc_entropy_list_normalize_base = np.abs(final_mc_entropy_list_normalize_base)
                final_mc_entropy_list_normalize_base[final_mc_entropy_list_normalize_base<10] = 10
                normalized_bias_per_state = bias / final_mc_entropy_list_normalize_base
                logger.store(NormQBias=normalized_bias_per_state)
                normalized_bias_sqr_per_state = bias_squared / final_mc_entropy_list_normalize_base
                logger.store(NormQBiasSqr=normalized_bias_sqr_per_state)

                if use_ave_q_in_analysis:
                    with torch.no_grad():
                        q_prediction_list = []
                        for q_i in range(num_Qs):
                            q_prediction_temp = q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
                            q_prediction_list.append(q_prediction_temp)
                        q_prediction_cat = torch.cat(q_prediction_list, dim=1)
                        q_prediction = torch.mean(q_prediction_cat, dim=1).cpu().numpy().reshape(-1)
                    bias = q_prediction - final_mc_entropy_list
                    bias_abs = np.abs(bias)
                    bias_squared = bias ** 2
                    logger.store(QPredAvg=q_prediction)
                    logger.store(QBiasAvg=bias)
                    logger.store(QBiasAbsAvg=bias_abs)
                    logger.store(QBiasSqrAvg=bias_squared)
                    final_mc_entropy_list_normalize_base = final_mc_entropy_list.copy()
                    final_mc_entropy_list_normalize_base = np.abs(final_mc_entropy_list_normalize_base)
                    final_mc_entropy_list_normalize_base[final_mc_entropy_list_normalize_base < 10] = 10
                    normalized_bias_per_state = bias / final_mc_entropy_list_normalize_base
                    logger.store(NormQBiasAvg=normalized_bias_per_state)
                    normalized_bias_sqr_per_state = bias_squared / final_mc_entropy_list_normalize_base
                    logger.store(NormQBiasSqrAvg=normalized_bias_sqr_per_state)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('CurrentG', G)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Std', with_min_and_max=True)
            logger.log_tabular('Alpha', with_min_and_max=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('PreTanh', with_min_and_max=True)
            logger.log_tabular('Time', time.time()-start_time)
            if do_analysis and do_new_analysis_scheme:
                logger.log_tabular("MCDisRet", with_min_and_max=True)
                logger.log_tabular("MCDisRetEnt", with_min_and_max=True)
                logger.log_tabular("QPred", with_min_and_max=True)
                logger.log_tabular("QBias", with_min_and_max=True)
                logger.log_tabular("QBiasAbs", with_min_and_max=True)
                logger.log_tabular("NormQBias", with_min_and_max=True)
                logger.log_tabular("QBiasSqr", with_min_and_max=True)
                logger.log_tabular("NormQBiasSqr", with_min_and_max=True)
                if use_ave_q_in_analysis:
                    logger.log_tabular("QPredAvg", with_min_and_max=True)
                    logger.log_tabular("QBiasAvg", with_min_and_max=True)
                    logger.log_tabular("QBiasAbsAvg", with_min_and_max=True)
                    logger.log_tabular("NormQBiasAvg", with_min_and_max=True)
                    logger.log_tabular("QBiasSqrAvg", with_min_and_max=True)
                    logger.log_tabular("NormQBiasSqrAvg", with_min_and_max=True)
            logger.dump_tabular()
            sys.stdout.flush()


    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    aqe(lambda: gym.make(args.env), hidden_sizes=[args.hid] * args.l,
         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
         steps_per_epoch=args.steps_per_epoch,
         logger_kwargs=logger_kwargs)


