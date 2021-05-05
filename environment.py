import numpy as np

print_game = False
N_ROW = 10
N_COL = 21
actions = [0,   # hit
           1]   # stick


class Card:

    def __init__(self):
        self.value = np.random.randint(10) + 1
        self.color = -1 if (np.random.randint(3) == 0) else 1

    def __str__(self):
        colorString = 'red' if self.color == -1 else 'black'
        return colorString + ' ' + str(self.value)


# state is a tuple with (sum, dealers first card) and action is a string 'hit' or 'stick'
def step(state, action):
    playerSum = state[0]+1
    dealerSum = state[1]+1
    done = False
    reward = 0

    if action == 0:     # hit
        newCard = Card()
        playerSum += newCard.color*newCard.value

        if print_game:
            print('player hit:')
            print('player draw:', newCard)
            print('player sum:', playerSum)

        if not 0 < playerSum < 22:
            done = True
            reward = -1

            if print_game:
                print('player bust')
                print('reward:', reward)
                print('------ player lose ------')

    if action == 1:     # stick
        done = True

        if print_game:
            print('player stick')

        while 0 < dealerSum < 17:
            newDealerCard = Card()
            dealerSum += newDealerCard.color*newDealerCard.value

            if print_game:
                print('dealer hit:')
                print('dealer draw:', newDealerCard)
                print('dealer sum:', dealerSum)

        if not 0 < dealerSum < 22:
            reward = 1

            if print_game:
                print('dealer bust')
                print('reward:', '1')
                print('<><><> player win <><><>')
        else:
            if dealerSum < playerSum:
                reward = 1
                if print_game:
                    print('player higher')
                    print('reward:', reward)
                    print('<><><> player win <><><>')

            elif dealerSum > playerSum:
                reward = -1

                if print_game:
                    print('dealer higher')
                    print('reward:', reward)
                    print('------ player lose ------')
            else:
                reward = 0

                if print_game:
                    print('draw')
                    print('reward:', reward)
                    print('...... draw ......')

    return (playerSum-1, state[1]), reward, done


def to_s(row, col):     # row = sum, col = dealer card
    return row * N_ROW + col


def init():

    s = (Card().value-1, Card().value-1)
    if print_game:
        print('player init card:')
        print(s[0]+1)
        print('dealer init card:')
        print(s[1]+1)

    return s
