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
SEED = para.SEED


def evaluate(env):
    eval_reward = []
    tcn = [0] * 6
    for i in range(1):
        step = 0
        obs = env.reset(md_list, bs_list)
        episode_reward = 0
        while True:
            obs = env.get_state()
            s_reward = 0
            for n in range(0, N):
                bs_list[n].res_F = F_BS[n]
            for m in range(0, M):
                b = agent.predict(obs[m])
                obs2 = np.append(obs[m], b / N)
                f = agent2.predict_e(obs2.astype('float32'))
                obs, reward, done, tcn, T1, E1, T1_pmax, E1_pmax, phi_p, phi_pmax, p = env.step_real(m, b, f, tcn, bs_list,
                                                                                                md_list, step)
                s_reward += reward
            episode_reward += s_reward
            step += 1
            env.reset_state()
            if step >= steps - 1:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


if __name__ == '__main__':
    es = []
    loss = []
    loss2 = []
    Alpha = []
    best_reward = 0
    Model_M = [20, 30, 40]
    for model_M in Model_M:
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
            algorithm.load_model('models/best_d3qn_real_model_{}.pth'.format(model_M))
            agent = D3QN.Agent(algorithm, obs_dim=obs_shape, act_dim=action_dim, e_greed=0.9,
                               e_greed_decrement=0.9 / (max_episode * steps * M))
            model2 = SAC.Model(obs_dim=obs_shape + 1, act_dim=action_dim2)
            actor_model = SAC.ActorModel(obs_dim=obs_shape + 1, act_dim=action_dim2)
            algorithm2 = SAC.SAC(model2, actor_model, gamma=SAC_GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
            algorithm2.load_model('models/best_sac_real_model_{}.pth'.format(model_M))
            agent2 = SAC.Agent(algorithm2, obs_dim=obs_shape + 1, act_dim=action_dim2)
            for n in range(0, N):
                bs = BS.BS(n, F_BS[n], random.uniform(0, 1600), random.uniform(0, 1200), 30)
                bs_list.append(bs)

            for m in range(0, M):
                md = MD.MD(m, F_MD, env, bs_list)
                md_list.append(md)

            e = []
            episode = 0
            total_reward = 0
            while episode < max_episode:
                episode += 1
                # test part
                eval_reward = evaluate(env)
                e.append(eval_reward)
            es.append(e)

        with open("result/RealD3SAC_{}_{}.pickle".format(model_M, M), 'wb') as f:
            pickle.dump(es, f)
