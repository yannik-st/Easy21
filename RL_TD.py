import environment as env
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from RL_MC import mc

N_STATES = 110
N_0 = 100
y = 0.95
N_EPISODES = 50000
l_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#l_value = 0.5

def f(X, Y, Q):
    V_ret = []
    x = X-1
    y = Y-1

    for i in range(0, len(x[:, 0])):
        for j in range(0, len(y[0, :])):
            V_ret.append(Q[x[i][j], y[i][j], np.argmax(Q[x[i][j], y[i][j]])])
    return np.reshape(np.array(V_ret), np.shape(x))


def main(argv=None):

    if argv is None:
        argv = sys.argv

    Q_star = mc()

    mse_list = []

    t0 = time.time()

    for l_value in l_values:
        Q = np.zeros((env.N_COL, env.N_ROW, len(env.actions)))

        n_s = np.zeros((env.N_COL, env.N_ROW))
        n_sa = np.zeros((env.N_COL, env.N_ROW, len(env.actions)))
        E = np.zeros((env.N_COL, env.N_ROW, len(env.actions)))

        for i_episodes in range(0, N_EPISODES):  # episode
            i_steps = 0

            done = False

            states = []
            actions = []
            rewards = []

            s = env.init()
            a = env.actions[np.random.randint(len(env.actions))]  # explore

            while not done:
                i_steps += 1

                s1, r, done = env.step(s, a)

                n_s[s] += 1
                n_sa[s][a] += 1
                E[s][a] += 1

                if not done:
                    e = N_0 / (N_0 + n_s[s])
                    if np.random.rand() < e:
                        a1 = env.actions[np.random.randint(len(env.actions))]  # explore
                    else:
                        a1 = np.argmax(Q[s1])  # exploit (act greedy)

                    delta = r + y * Q[s1][a1] - Q[s][a]
                else:
                    delta = r - Q[s][a]

                states.append(s)
                actions.append(a)
                rewards.append(r)

                for k in range(0, len(states)):
                    s_k = states[k]
                    a_k = actions[k]

                    ss = 1 / n_sa[s_k][a_k]
                    Q[s_k][a_k] = Q[s_k][a_k] + ss*delta*E[s_k][a_k]
                    E[s_k][a_k] = y*l_value*E[s_k][a_k]

                if not done:
                    s = s1
                    a = a1

        #    if eval_counter == N_eval:
        #        eval_rewards.append(eval_sum / N_eval)
        #        eval_sum = 0
        #        eval_counter = 0

        mse = 0
        for i in range(0, env.N_COL):
            for j in range(0, env.N_ROW):
                for k in range(0, len(env.actions)):
                    mse += (Q[i][j][k]-Q_star[i][j][k])**2

        print('elapsed time:', time.time() - t0)
        print('l-Value: ', l_value)
        print('MSE: ', mse)
        mse_list.append(mse)

    print(mse_list)
    plt.plot(np.array(mse_list))
    plt.show()


        # print(len(eval_rewards))
        #plt.plot(eval_rewards)
        #plt.show()

    '''    x_player = np.array(range(1, 22))
        x_dealer = np.array(range(1, 11))
        X_player, X_dealer = np.meshgrid(x_player, x_dealer)
    
        Z = f(X_player, X_dealer, Q)
    
        ax = plt.axes(projection='3d')
        # ax.plot_wireframe(X_player, X_dealer, Z)
        ax.plot_surface(X_player, X_dealer, Z, cmap=cm.inferno,
                        linewidth=0, antialiased=False)
        ax.set_xlabel('player sum')
        ax.set_ylabel('dealer showing')
        ax.set_zlabel('value')
        plt.show()'''


if __name__ == '__main__':
    main()
