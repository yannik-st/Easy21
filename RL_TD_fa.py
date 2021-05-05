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
l_values = [0]
e = 0.05
ss = 0.01


def f(X, Y, Q):
    V_ret = []
    x = X-1
    y = Y-1

    for i in range(0, len(x[:, 0])):
        for j in range(0, len(y[0, :])):
            V_ret.append(Q[x[i][j], y[i][j], np.argmax(Q[x[i][j], y[i][j]])])
    return np.reshape(np.array(V_ret), np.shape(x))


def phi(s, a):
    player_sum = [
        (1 <= s[0]+1 <= 6),
        (4 <= s[0]+1 <= 9),
        (7 <= s[0]+1 <= 12),
        (10 <= s[0]+1 <= 15),
        (13 <= s[0]+1 <= 18),
        (16 <= s[0]+1 <= 21)
    ]
    dealer_first = [
        (1 <= s[1]+1 <= 4),
        (4 <= s[1]+1 <= 7),
        (7 <= s[1]+1 <= 10)
    ]
    action = [
        (a == 0),
        a
    ]

    return np.reshape(np.outer(np.outer(np.array(player_sum), np.array(dealer_first)), np.array(action)), newshape=36)


def q_hat(s, a, theta):
    return np.dot(phi(s,a), theta)


def main(argv=None):

    if argv is None:
        argv = sys.argv

    Q_star = mc()

    theta = np.random.randn(36, 1)

    print('mc_done')

    mse_lambda_list = []
    mse_episode_list = []

    t0 = time.time()

    for l_value in l_values:
        E = np.zeros_like(theta)

        for i_episodes in range(0, N_EPISODES):  # episode
            i_steps = 0

            done = False

            states = []
            actions = []
            rewards = []

            s = env.init()
            a = env.actions[np.random.randint(len(env.actions))]  # explore

            mse = 0
            for i in range(0, env.N_COL):
                for j in range(0, env.N_ROW):
                    for k in range(0, len(env.actions)):
                        mse += (q_hat((i, j), k, theta) - Q_star[i][j][k]) ** 2
            mse_episode_list.append(mse)

            while not done:
                i_steps += 1

                s1, r, done = env.step(s, a)

                if not done:
                    if np.random.rand() < e:
                        a1 = env.actions[np.random.randint(len(env.actions))]  # explore
                    else:
                        a1 = np.argmax( [q_hat(s1, a_opt, theta) for a_opt in env.actions] )  # exploit (act greedy)

                    delta = r + y * q_hat(s1, a1, theta) - q_hat(s, a, theta)
                else:
                    delta = r - q_hat(s,a,theta)

                states.append(s)
                actions.append(a)
                rewards.append(r)

                for k in range(0, len(states)):
                    s_k = states[k]
                    a_k = actions[k]

                    E = y*l_value*E + phi(s_k, a_k).reshape(-1,1)
                    d_theta = ss*delta*E
                    theta += d_theta

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
                    mse += (q_hat((i,j),k, theta)-Q_star[i][j][k])**2

        print('elapsed time:', time.time() - t0)
        print('l-Value: ', l_value)
        print('MSE: ', mse)
        mse_lambda_list.append(mse)

    print(mse_lambda_list)
    plt.plot(np.array(mse_lambda_list))
    plt.show()

    plt.plot(np.array(mse_episode_list))
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
