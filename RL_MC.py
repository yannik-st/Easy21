import environment as env
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm

N_STATES = 110
N_0 = 100
y = 0.95
N_EPISODES = 500000
N_eval = 1000  # for evaluation we always look at the avg. reward of N_eval episodes


def f(X, Y, Q):
    V_ret = []
    x = X-1
    y = Y-1

    for i in range(0, len(x[:, 0])):
        for j in range(0, len(y[0, :])):
            V_ret.append(Q[x[i][j], y[i][j], np.argmax(Q[x[i][j], y[i][j]])])
    return np.reshape(np.array(V_ret), np.shape(x))


def mc():
    t0 = time.time()

    Q = np.zeros((env.N_COL, env.N_ROW, len(env.actions)))

    n_s = np.zeros((env.N_COL, env.N_ROW))
    n_sa = np.zeros((env.N_COL, env.N_ROW, len(env.actions)))

    eval_rewards = []
    eval_counter = 0
    eval_sum = 0

    for i in range(0, N_EPISODES):
        j = 0
        eval_counter += 1

        done = False

        states = []
        actions = []
        rewards = []

        s = env.init()
        e = N_0 / (N_0 + n_s[s])

        while not done:
            j += 1

            if np.random.rand() < e:
                a = np.random.randint(2)  # explore
            else:
                a = np.argmax(Q[s])  # exploit (act greedy)

            s1, r, done = env.step(s, a)

            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = s1

        G = 0
        k = 0
        for r in rewards:
            G += (y ** k) * r
            k += 1

        for k in range(0, len(states)):
            s = states[k]
            a = actions[k]

            n_s[s] += 1
            n_sa[s][a] += 1

            ss = 1 / n_sa[s][a]
            Q[s][a] = Q[s][a] + ss * (G - Q[s][a])

        eval_sum += G

        if eval_counter == N_eval:
            eval_rewards.append(eval_sum / N_eval)
            eval_sum = 0
            eval_counter = 0

    print('elapsed time:', time.time() - t0)
    # print(len(eval_rewards))
    '''plt.plot(eval_rewards)
    plt.show()

    x_player = np.array(range(1, 22))
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

    return Q


def main(argv=None):
    if argv is None:
        argv = sys.argv

    mc()


if __name__ == '__main__':
    main()
