import D3QN
import SAC
import MD
import BS
import ReplayMemory
import EdgeEnv
import para
import random
import numpy as np
import torch
import warnings
import pickle
warnings.filterwarnings("ignore", category=Warning)
LEARN_FREQ = para.LEARN_FREQ
SAC_LEARN_FREQ = para.SAC_LEARN_FREQ
MEMORY_SIZE = para.MEMORY_SIZE
SAC_MEMORY_SIZE = para.SAC_MEMORY_SIZE
MEMORY_WARMUP_SIZE = para.MEMORY_WARMUP_SIZE
SAC_MEMORY_WARMUP_SIZE = para.SAC_MEMORY_WARMUP_SIZE
BATCH_SIZE = para.BATCH_SIZE
SAC_BATCH_SIZE = para.SAC_BATCH_SIZE
LEARNING_RATE = para.LEARNING_RATE
ACTOR_LR = para.ACTOR_LR
CRITIC_LR = para.CRITIC_LR
DQN_GAMMA = para.DQN_GAMMA
SAC_GAMMA = para.SAC_GAMMA
TAU = para.TAU
NOISE = para.NOISE
N = para.N
M = para.M
w_t = para.w_t
w_e = para.w_e
F_BS = para.F_BS
F_MD = para.F_MD
steps = para.steps
zeta = para.zeta
A_max = para.A_max
max_episode = para.max_episode

alpha = para.alpha
T_list = []
E_list = []
res_F1 = []
res_F2 = []
res_F3 = []
TCN = []
TCN1 = []
TCN2 = []
TCN3 = []
SEED = para.SEED


def run_episode(env, rpm, rpm2):
    total_reward = 0
    obs = env.reset(md_list, bs_list)
    step = 0
    a = []
    while True:
        l = []
        l2 = []
        s_reward = 0
        obs = env.get_state()
        for n in range(0, N):
            bs_list[n].res_F = F_BS[n]
        for m in range(0, M):
            b = agent.sample(obs[m])
            preobs = obs[m]
            # if b == 0:
            #     f = 0
            #     obs, reward, done, _, _, _, _, _, _, _, _ = env.step_real(m, b, f, [0] * 6, bs_list, md_list, step)
            #     if m == M - 1:
            #         rpm.append((preobs, b, reward, obs[0], done))
            #     else:
            #         rpm.append((preobs, b, reward, obs[m + 1], done))
            #     s_reward += reward
            # else:
            obs2 = np.append(obs[m], b / N)
            f = agent2.predict(obs2.astype('float32'))
            obs, reward, done, _, _, _, _, _, _, _, _ = env.step_real(m, b, f, [0] * 6, bs_list, md_list, step)
            if m == M - 1:
                next_b = agent.predict(obs[0])
                next_obs2 = np.append(obs[0], next_b / N)
                rpm.append((preobs, b, reward, obs[0], done))
                rpm2.append((obs2, f, reward, next_obs2, done))
            else:
                next_b = agent.predict(obs[m + 1])
                next_obs2 = np.append(obs[m + 1], next_b / N)
                rpm.append((preobs, b, reward, obs[m + 1], done))
                rpm2.append((obs2, f, reward, next_obs2, done))
            s_reward += reward
            # train model
            if (len(rpm) > MEMORY_WARMUP_SIZE) and (m % LEARN_FREQ == 0):
                (batch_obs, batch_action, batch_reward, batch_next_obs,
                 batch_done) = rpm.sample(BATCH_SIZE)
                train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                         batch_next_obs,
                                         batch_done)  # s,a,r,s',done
                l.append(train_loss)
            if len(rpm2) > SAC_MEMORY_WARMUP_SIZE and (m % SAC_LEARN_FREQ) == 0:
                (batch_obs, batch_action, batch_reward, batch_next_obs,
                 batch_done) = rpm2.sample(SAC_BATCH_SIZE)
                SAC_train_loss, alpha = agent2.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                                                     batch_done)
                l2.append(SAC_train_loss)
                a.append(alpha.detach().numpy())
        step += 1
        env.reset_state()
        total_reward += s_reward
        if step >= steps - 1:
            break
    loss.append(np.mean(l))
    loss2.append(np.mean(l2))
    Alpha.append(np.mean(a))
    return total_reward


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env):
    eval_reward = []
    eval_T1 = []
    eval_E1 = []
    eval_T1_pmax = []
    eval_E1_pmax = []
    eval_phi_p = []
    eval_phi_pmax = []
    eval_p = []
    tcn = [0] * 6
    for i in range(5):
        step = 0
        obs = env.reset(md_list, bs_list)
        episode_reward = 0
        episode_T1 = 0
        episode_E1 = 0
        episode_T1_pmax = 0
        episode_E1_pmax = 0
        episode_phi_p = 0
        episode_phi_pmax = 0
        episode_p = 0
        while True:
            obs = env.get_state()
            s_reward = 0
            sT1 = 0
            sE1 = 0
            sT1_pmax = 0
            sE1_pmax = 0
            sphi_p = 0
            sphi_pmax = 0
            sp = 0
            for n in range(0, N):
                bs_list[n].res_F = F_BS[n]
            for m in range(0, M):
                b = agent.predict(obs[m])
                obs2 = np.append(obs[m], b / N)
                f = agent2.predict_e(obs2.astype('float32'))
                obs, reward, done, tcn, T1, E1, T1_pmax, E1_pmax, phi_p, phi_pmax, p = env.step_real(m, b, f, tcn, bs_list,
                                                                                                md_list, step)
                s_reward += reward
                sT1 += T1
                sE1 += E1
                sT1_pmax += T1_pmax
                sE1_pmax += E1_pmax
                sphi_p += phi_p
                sphi_pmax += phi_pmax
                sp += p
            res_F1.append(bs_list[0].res_F)
            res_F2.append(bs_list[1].res_F)
            res_F3.append(bs_list[2].res_F)
            episode_reward += s_reward
            episode_T1 += sT1
            episode_E1 += sE1
            episode_T1_pmax += sT1_pmax
            episode_E1_pmax += sE1_pmax
            episode_phi_p += sphi_p
            episode_phi_pmax += sphi_pmax
            episode_p += sp
            step += 1
            env.reset_state()
            if step >= steps - 1:
                break
        eval_reward.append(episode_reward)
        eval_T1.append(episode_T1)
        eval_E1.append(episode_E1)
        eval_T1_pmax.append(episode_T1_pmax)
        eval_E1_pmax.append(episode_E1_pmax)
        eval_phi_p.append(episode_phi_p)
        eval_phi_pmax.append(episode_phi_pmax)
        eval_p.append(episode_p)
    TCN1.append(tcn[0] / tcn[3])
    TCN2.append(tcn[1] / tcn[4])
    TCN3.append(tcn[2] / tcn[5])
    TCN.append(sum([tcn[0], tcn[1], tcn[2]]) / sum([tcn[3], tcn[4], tcn[5]]))
    return np.mean(eval_reward), np.mean(eval_T1), np.mean(eval_E1), np.mean(eval_T1_pmax), np.mean(
        eval_E1_pmax), np.mean(eval_phi_p), np.mean(eval_phi_pmax), np.mean(eval_p)


if __name__ == '__main__':
    es = []
    ts = []
    es_T1 = []
    es_E1 = []
    es_T1_pmax = []
    es_E1_pmax = []
    es_phi_p = []
    es_phi_pmax = []
    es_p = []
    loss = []
    loss2 = []
    Alpha = []
    best_reward = 0
    for seed in SEED:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        env = EdgeEnv.EdgeEnv()
        action_dim = env.action_space.n
        obs_shape = env.observation_space.shape[1]
        action_dim2 = env.action_space2.shape[0]
        bs_list = []
        md_list = []
        model = D3QN.Model(obs_dim=obs_shape, act_dim=action_dim)
        algorithm = D3QN.DDQN(model, act_dim=action_dim, gamma=DQN_GAMMA, lr=LEARNING_RATE)
        agent = D3QN.Agent(algorithm, obs_dim=obs_shape, act_dim=action_dim, e_greed=0.9,
                           e_greed_decrement=0.9 / (max_episode * steps * M))
        model2 = SAC.Model(obs_dim=obs_shape + 1, act_dim=action_dim2)
        actor_model = SAC.ActorModel(obs_dim=obs_shape + 1, act_dim=action_dim2)
        algorithm2 = SAC.SAC(model2, actor_model, gamma=SAC_GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
        agent2 = SAC.Agent(algorithm2, obs_dim=obs_shape + 1, act_dim=action_dim2)
        for n in range(0, N):
            bs = BS.BS(n, F_BS[n], random.uniform(0, 1600), random.uniform(0, 1200), 30)
            bs_list.append(bs)

        for m in range(0, M):
            md = MD.MD(m, F_MD, env, bs_list)
            md_list.append(md)

        rpm = ReplayMemory.ReplayMemory(MEMORY_SIZE)
        rpm2 = ReplayMemory.ReplayMemory(SAC_MEMORY_SIZE)

        while len(rpm) < MEMORY_WARMUP_SIZE:
            run_episode(env, rpm, rpm2)
        e = []
        t = []
        e_T1 = []
        e_E1 = []
        e_T1_pmax = []
        e_E1_pmax = []
        e_phi_p = []
        e_phi_pmax = []
        e_p = []
        episode = 0
        total_reward = 0
        while episode < max_episode:
            # train part
            for i in range(0, 1):
                total_reward = run_episode(env, rpm, rpm2)
                episode += 1
            t.append(total_reward)
            # test part
            eval_reward, eval_T1, eval_E1, eval_T1_pmax, eval_E1_pmax, eval_phi_p, eval_phi_pmax, eval_p = evaluate(env)
            if eval_reward > best_reward:
                best_reward = eval_reward
                agent.alg.save_model('models/best_d3qn_real_model_{}.pth'.format(M))
                agent2.alg.save_model('models/best_sac_real_model_{}.pth'.format(M))
            e.append(eval_reward)
            e_T1.append(eval_T1)
            e_E1.append(eval_E1)
            e_T1_pmax.append(eval_T1_pmax)
            e_E1_pmax.append(eval_E1_pmax)
            e_phi_p.append(eval_phi_p)
            e_phi_pmax.append(eval_phi_pmax)
            e_p.append(eval_p)

        es.append(e)
        ts.append(t)
        es_T1.append(e_T1)
        es_E1.append(e_E1)
        es_T1_pmax.append(e_T1_pmax)
        es_E1_pmax.append(e_E1_pmax)
        es_phi_p.append(e_phi_p)
        es_phi_pmax.append(e_phi_pmax)
        es_p.append(e_p)

    with open("result/pytorch_md3qn_mSAC_p_real_t_{}.pickle".format(M), 'wb') as f:
        pickle.dump(ts, f)
    with open("result/pytorch_md3qn_mSAC_p_real_e_{}.pickle".format(M), 'wb') as f:
        pickle.dump(es, f)
    with open("result/pytorch_md3qn_mSAC_p_real_e_T1_{}.pickle".format(M), 'wb') as f:
        pickle.dump(es_T1, f)
    with open("result/pytorch_md3qn_mSAC_p_real_e_E1_{}.pickle".format(M), 'wb') as f:
        pickle.dump(es_E1, f)
    with open("result/pytorch_md3qn_mSAC_p_real_e_T1_pmax_{}.pickle".format(M), 'wb') as f:
        pickle.dump(es_T1_pmax, f)
    with open("result/pytorch_md3qn_mSAC_p_real_e_E1_pmax_{}.pickle".format(M), 'wb') as f:
        pickle.dump(es_E1_pmax, f)
    with open("result/pytorch_md3qn_mSAC_p_real_e_phi_p_{}.pickle".format(M), 'wb') as f:
        pickle.dump(es_phi_p, f)
    with open("result/pytorch_md3qn_mSAC_p_real_e_phi_pmax_{}.pickle".format(M), 'wb') as f:
        pickle.dump(es_phi_pmax, f)
    with open("result/pytorch_md3qn_mSAC_p_real_e_p_{}.pickle".format(M), 'wb') as f:
        pickle.dump(es_p, f)
    with open("result/pytorch_md3qn_mSAC_p_real_e_loss_{}.pickle".format(M), 'wb') as f:
        pickle.dump(loss, f)
    with open("result/pytorch_md3qn_mSAC_p_real_e_loss2_{}.pickle".format(M), 'wb') as f:
        pickle.dump(loss2, f)
    with open("result/pytorch_md3qn_mSAC_p_real_e_alpha_{}.pickle".format(M), 'wb') as f:
        pickle.dump(Alpha, f)
    if M == 20:
        with open("result/pytorch_md3qn_mSAC_real_p_TCN1.pickle", 'wb') as f:
            pickle.dump(TCN1, f)
        with open("result/pytorch_md3qn_mSAC_real_p_TCN2.pickle", 'wb') as f:
            pickle.dump(TCN2, f)
        with open("result/pytorch_md3qn_mSAC_real_p_TCN3.pickle", 'wb') as f:
            pickle.dump(TCN3, f)
        with open("result/pytorch_md3qn_mSAC_real_p_TCN.pickle", 'wb') as f:
            pickle.dump(TCN, f)
        with open("result/pytorch_md3qn_mSAC_real_p_res_F1.pickle", 'wb') as f:
            pickle.dump(res_F1, f)
        with open("result/pytorch_md3qn_mSAC_real_p_res_F2.pickle", 'wb') as f:
            pickle.dump(res_F2, f)
        with open("result/pytorch_md3qn_mSAC_real_p_res_F3.pickle", 'wb') as f:
            pickle.dump(res_F3, f)
