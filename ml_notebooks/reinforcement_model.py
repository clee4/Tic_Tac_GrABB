#%%
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.models import load_model
from keras import optimizers
import random
import numpy as np
import math
import os

def one_hot(state):
	current_state = []

	for square in state:
		if square == 0:
			current_state.append(1)
			current_state.append(0)
			current_state.append(0)
		elif square == 1:
			current_state.append(0)
			current_state.append(1)
			current_state.append(0)
		elif square == -1:
			current_state.append(0)
			current_state.append(0)
			current_state.append(1)
    
	return current_state

def get_outcome(state):
	total_reward = 0

	if (state[0] == state[1] == state[2]) and not state[0] == 0:
		total_reward = state[0]	
	elif (state[3] == state[4] == state[5]) and not state[3] == 0:
		total_reward = state[3]	
	elif (state[6] == state[7] == state[8]) and not state[6] == 0:
		total_reward = state[6]	
	elif (state[0] == state[3] == state[6]) and not state[0] == 0:
		total_reward = state[0]	
	elif (state[1] == state[4] == state[7]) and not state[1] == 0:
		total_reward = state[1]	
	elif (state[2] == state[5] == state[8]) and not state[2] == 0:
		total_reward = state[2]	
	elif (state[0] == state[4] == state[8]) and not state[0] == 0:
		total_reward = state[0]	
	elif (state[2] == state[4] == state[6]) and not state[2] == 0:
		total_reward = state[2]

	return total_reward

def process_games(games, model, model_num, file_name, reward_dep=.7):
    xt = 0
    ot = 0
    dt = 0
    states = []
    q_values = []

    for game in games:
        total_reward = get_outcome(game[len(game) - 1])
        if total_reward == -1:
            ot += 1
        elif total_reward == 1:
            xt += 1
        else:
            dt += 1

        for i in range(model_num-1, len(game)-1, 2):
            for j in range(0, 9):
                if not game[i][j] == game[i + 1][j]:
                    reward_vector = np.zeros(9)
                    reward_vector[j] = total_reward*(reward_dep**(math.floor((len(game) - i) / 2) - 1))
                    # print(reward_vector)
                    states.append(game[i].copy())
                    q_values.append(reward_vector.copy())

    zipped = list(zip(states, q_values))
    random.shuffle(zipped)
    states, q_values = zip(*zipped)
    new_states = []
    for state in states:
        new_states.append(one_hot(state))

    model.fit(np.asarray(new_states), np.asarray(q_values), epochs=2, 
              batch_size=len(q_values), verbose=1)

    path = os.path.abspath('')
    path = os.path.join(path, 'storage')
    path = os.path.join(path, file_name)
    model.save(path)
    del model
    model = load_model(path)
    print(xt/20, ot/20, dt/20)

    return model

def select_random_move(model, board):
    while True:
        c = random.randint(0, 8)
        if board[c] == 0:
            return c
            

def select_smart_move(model, board):
    pre = model.predict(np.asarray([one_hot(board)]), batch_size=1)[0]
    highest = -1000
    num = -1
    for j in range(9):
        if board[j] == 0 and pre[j] > highest:
                highest = pre[j].copy()
                num = j
    
    return num

def play_and_train(model, model_2, games_until_training=1000, total_games=8000, e_greedy=.7):
    board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    games = []
    current_game = []

    for i in range(0, total_games):
        playing = True
        nn_turn = True
        board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        # sides --> -1 = Os, 1 = Xs
        current_game = []
        current_game.append(board.copy())

        while playing:
            if nn_turn:
                if random.uniform(0, 1) <= e_greedy:
                    board[select_random_move(model,board)]=1
                    current_game.append(board.copy())
                else:
                    board[select_smart_move(model, board)] = 1
                    current_game.append(board.copy())

            else:
                if random.uniform(0, 1) <= e_greedy:
                    board[select_random_move(model_2,board)] = -1
                    current_game.append(board.copy())
                else:
                    board[select_smart_move(model_2, board)] = -1
                    current_game.append(board.copy())

            playable = False

            for square in board:
                if square == 0:
                    playable = True

            if not get_outcome(board) == 0:
                playable = False

            if not playable:
                playing = False

            nn_turn = not nn_turn

        games.append(current_game)
    
        if (i+1)%games_until_training == 0:
            print('\nGEN %d'%((i+1)/games_until_training))
            print('MODEL 1')
            model = process_games(games, model, 1, 'tic_tac_toe.h5')

            print('MODEL 2')
            model_2 = process_games(games, model_2, 2, 'tic_tac_toe_2.h5')

            games = []

    return (model, model_2)
#%%
model = Sequential()
model.add(Dense(units=130, activation='relu', input_dim=27, kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(units=250, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(units=140, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(units=60, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(9, kernel_initializer='random_uniform', bias_initializer='zeros'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model_2 = Sequential()
model_2.add(Dense(units=130, activation='relu', input_dim=27, kernel_initializer='random_uniform', bias_initializer='zeros'))
model_2.add(Dense(units=250, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model_2.add(Dense(units=140, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model_2.add(Dense(units=60, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model_2.add(Dense(9, kernel_initializer='random_uniform', bias_initializer='zeros'))
model_2.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

#%%
try:
	model = load_model('tic_tac_toe.h5')
	model_2 = load_model('tic_tac_toe_2.h5')
	print('Pre-existing model found... loading data.')
except:
	pass
model, model_2 = play_and_train(model, model_2)

#%%
file_name = 'asdf.h5'
path = os.path.abspath('')
path = os.path.join(path, 'storage')
path = os.path.join(path, file_name)
print(path)

#%%
def new_one_hot(state):
    current_state = []

    for position in state:
        if position[-1] == ' ':
            current_state.append(1)
            current_state.append(0)
            current_state.append(0)
        elif position[-1] == 'x':
            current_state.append(0)
            current_state.append(1)
            current_state.append(0)
        elif position[-1] == 'o':
            current_state.append(0)
            current_state.append(0)
            current_state.append(1)

    return current_state

#%%
board_nums = [0,1,-1,0,0,0,0,0,0]
board_norm = [[' '], ['x'], ['o'],[' '],[' '],[' '],[' '],[' '],[' ']]

#%%
print(one_hot(board_nums))
print(new_one_hot(board_norm))

#%%
