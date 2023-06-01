"""
    This file contains your final_strategy that will be submitted to the contest.
    It will only be run on your local machine, so you can import whatever you want!
    Remember to supply a unique PLAYER_NAME or your submission will not succeed.
"""

import numpy as np
import copy
from dice import make_fair_dice
from compare_strategies import compare

PLAYER_NAME = 'okonoko'  # Change this line!
GOAL_SCORE = 100
MAX_DICE = 10

cached_full_score_guidline = []
def final_strategy(score, opponent_score):

    global cached_full_score_guidline # Needed to modify global copy of globvar
    full_score_guidline = cached_full_score_guidline if cached_full_score_guidline else read()
    cached_full_score_guidline = full_score_guidline

    best_num_dice = 0
    best_win_chance = 0
    for num_dice in range(0, MAX_DICE+1):
        win_chance = full_score_guidline[0][11][min(score, GOAL_SCORE)][min(opponent_score, GOAL_SCORE)][num_dice+1]

        if win_chance > best_win_chance:
            best_win_chance = win_chance
            best_num_dice = num_dice
    return best_num_dice

def create_optimal_solutions():
    full_score_guidline = reset()

    # for eight sided dice
    full_score_guidline[1][0] = copy.deepcopy(creator(full_score_guidline[1][0], full_score_guidline[1][0], turn=0, six_sided=False))
    for i in range(1, MAX_DICE+2):
        print("eight sided", i)
        full_score_guidline[1][i] = copy.deepcopy(full_score_guidline[1][0])


    # for six sided dice
    for i in range(MAX_DICE+1, -1, -1):
        print("six sided", i)
        full_score_guidline[0][i] = copy.deepcopy(creator(full_score_guidline[0][i], full_score_guidline[0][min(i+1,MAX_DICE+1)], turn=i, six_sided=True, eight_sided_next_turn_score_guidline=full_score_guidline[1][min(i+1,MAX_DICE+1)]))

    write(full_score_guidline)

def creator(score_guidline, next_turn_score_guidline, turn=11, six_sided=True, eight_sided_next_turn_score_guidline=None):
    def iterate(I, J, isI):
        if I == -1 or J == -1:
            return
        if isI:
            for i in range(I, -1, -1):
                score_guidline[i][J] = copy.deepcopy(define_strategy(i, J, next_turn_score_guidline, turn, six_sided, eight_sided_next_turn_score_guidline))
            iterate(I,J-1, not isI)
        else:
            for j in range(J, -1, -1):
                score_guidline[I][j] = copy.deepcopy(define_strategy(I, j, next_turn_score_guidline, turn, six_sided, eight_sided_next_turn_score_guidline))
            iterate(I-1,J, not isI)

    iterate(GOAL_SCORE-1, GOAL_SCORE-1, False)
    # print(turn)
    # print(score_guidline)
    return score_guidline

def define_strategy(i, j, score_guidline, turn=11, six_sided=True, eight_sided_next_turn_score_guidline=None):
    greatest_average_win_chance = 0
    win_chance_by_num_rolls = [0]*(MAX_DICE+1)

    dice_sides = 6 if six_sided else 8
    for k in range(0, MAX_DICE+1):
        num_to_check = dice_sides**k
        possible_rolls = max(dice_sides*(k-1), 1)
        total_win_chance = 0.0
        permutation_sum = 0
        for l in range(k, possible_rolls):
            # returns 
            if k == 0:
                new_value = piggy_points(j)
                permutations = 1
                permutation_sum+=permutations
            else:
                new_value = l + k
                permutations = get_permutations_with_sum(l, dice_sides-1, k)
                permutation_sum+=permutations


            extra_turn = detect_more_boar(i, j, new_value)
            # extra turn
            if six_sided == True and (extra_turn or (turn == k and turn < 11)):
                # if permutations*win_chance_from_guidline(i+new_value, j, eight_sided_next_turn_score_guidline, 0) < permutations*win_chance_from_guidline(j, i+new_value, eight_sided_next_turn_score_guidline, 1)  :
                #     print('aiyaa')

                total_win_chance += permutations*win_chance_from_guidline(i+new_value, j, eight_sided_next_turn_score_guidline, 0)            
            else:
                # note i an j reversed
                total_win_chance += permutations*win_chance_from_guidline(j, i+new_value, score_guidline, 1)            

        if six_sided and permutation_sum > (dice_sides-1)**k:
            print(k, 'problem1')
        
        if k != 0:
            new_value = 1
            extra_turn = detect_more_boar(i, j, new_value)
            permutations = (1-((dice_sides-1)/dice_sides)**k)*num_to_check
            permutation_sum+=permutations
            # extra turn
            if six_sided == True and (extra_turn or (turn == k and turn < 11)):
                # note i an j reversed
                total_win_chance += permutations*win_chance_from_guidline(i+new_value, j, eight_sided_next_turn_score_guidline, 0)            
            else:
                # note i an j reversed
                total_win_chance += permutations*win_chance_from_guidline(j, i+new_value, score_guidline, 1)   

        #print(permutation_sum, num_to_check)
        # if permutation_sum > num_to_check:
        #     print('problem2')

        average_win_chance = total_win_chance/num_to_check
        # if average_win_chance == 1:
        #     return [1, k]
        win_chance_by_num_rolls[k] = average_win_chance
        if average_win_chance > greatest_average_win_chance:
            greatest_average_win_chance = average_win_chance
    if i==50 and j ==50 or i==20 and j ==80:
        print([greatest_average_win_chance] + win_chance_by_num_rolls)
    return [greatest_average_win_chance] + list(win_chance_by_num_rolls)

# turn = 0 if it is player turn after my last role
# turn = 1 if it is opponent turn after my last role
def win_chance_from_guidline(i, j, score_guidline, turn):
    i = min(i, GOAL_SCORE)
    j = min(j, GOAL_SCORE)
    #print(i, j)
    #print(score_guidline[i][j])
    if score_guidline[i][j][0] == -1:
        print('algorithm issue: win chance calculation')
    return abs(turn-score_guidline[i][j][0])

# CURRENTLY NOT IN USE
# returns dice_total
def calculate_dice_total(num_rolls, index, opponent_score, dice_sides):
    if num_rolls == 0:
        return piggy_points(opponent_score)

    total = 0
    for _ in range(0, num_rolls):
        dice_roll_value = index%6+1
        if dice_roll_value == 1:
            return 1
        total += dice_roll_value
        index = index//dice_sides
    return total

def detect_more_boar(player_score, opponent_score, new_value):
    return more_boar(player_score + new_value, opponent_score)

# @whuber 
def combinations(n, k):
	total_combination = 1
	for i in range(0, k):
		total_combination = total_combination * ((n - i) / (i + 1))

	return total_combination

def get_permutations_with_sum(diceSum, sides, rolls):
    x = sides
    m = diceSum
    n = rolls
    mMinusN = m - n
    perms = 0
	
    k = 0
    j = mMinusN
    while j >= 0:
        perms += combinations(n, k) * combinations(-n, j) * pow(-1, j + k)
        k+=1
        j-=x

    return round(perms)


# storage structure
# - for max dice the intuition is that it is the maximum number of dice that can be rolled. 
#   The parent list is of size Max_dice*2 because we need to handle different cases for
#   each turn due to the time trot rule. Turn Max_dice+1 is the case that should be used if
#   the turn number is greater than the maximum number of dice
# - (my win chance, best num rolls)


###### saving logic

# reset
def reset():
    score_guidline = [-1]+list_multiplication(-1,(MAX_DICE+1))
    score_guidline2 = [[-1]+[-1]*(MAX_DICE+1)]

    score_guidline = list_multiplication(list_multiplication(score_guidline, (GOAL_SCORE)), (GOAL_SCORE))
    score_guidline2 = [score_guidline2 * (GOAL_SCORE)]*(GOAL_SCORE)

    score_guidline += [list_multiplication([1]+list_multiplication(-1, (MAX_DICE+1)), GOAL_SCORE)]
    score_guidline2 += [([[1]+[-1]*(MAX_DICE+1)]*(GOAL_SCORE))]

    score_guidline = [x + [[0]+list_multiplication(-1, (MAX_DICE+1))] for x in score_guidline]
    score_guidline2 = [x + [[0]+[-1]*(MAX_DICE+1)] for x in score_guidline2]

    score_guidline = list_multiplication(list_multiplication(score_guidline, (MAX_DICE+2)), 2)
    score_guidline2 = [[score_guidline2]*(MAX_DICE+2)]*2

    write(score_guidline)
    return score_guidline

def list_multiplication(copy_list, multiplyBy):
    return [copy_list for _ in range(0, multiplyBy)]

def write(score_guidline):
    open('test.npy', 'w').close()
    score_guidline = np.asarray(score_guidline, dtype=np.float32)
    np.save('test.npy', score_guidline)

def read(filename = 'test.npy'):
    new_data = np.load('test.npy')
    new_data = new_data.reshape((2,MAX_DICE+2,GOAL_SCORE+1,GOAL_SCORE+1,MAX_DICE+2))
    return new_data.tolist()

#[[[0, 0], [0, 0], [1, 0]], [[1, 0], [1, 0], [0, 0]], [[0, 0], [0, 0], [1, 0]], [[1, 1], [1, 1], [0, 1]]]
#[[[1, 1], [1, 1], [0, 0]], [[1, 0], [1, 0], [1, 1]], [[0, 0], [0, 0], [1, 0]], [[1, 1], [1, 1], [0, 1]]]


# Functions from Hog
def piggy_points(score):
    """Return the points scored from rolling 0 dice.

    score:  The opponent's current score.
    """
    # BEGIN PROBLEM 2
    sqrd_score = score ** 2
    smallest_digit = sqrd_score%10
    while sqrd_score > 0:
        cur_digit = sqrd_score % 10
        if smallest_digit > cur_digit:
            smallest_digit = cur_digit
        sqrd_score = sqrd_score // 10

    return smallest_digit + 3 
    # END PROBLEM 2

def more_boar(player_score, opponent_score):
    """Return whether the player gets an extra turn.

    player_score:   The total score of the current player.
    opponent_score: The total score of the other player.

    >>> more_boar(21, 43)
    True
    >>> more_boar(22, 43)
    True
    >>> more_boar(43, 21)
    False
    >>> more_boar(12, 12)
    False
    >>> more_boar(7, 8)
    False
    """
    # BEGIN PROBLEM 4

    def leftest_digit(num, index=1):
        leftest_num = num 
        
        while leftest_num >=10**index:
            leftest_num = leftest_num//10
        
        leftest_num = leftest_num % 10
             
        return leftest_num if num >=10 or index > 1 else 0

        
    return leftest_digit(player_score, 1) < leftest_digit(opponent_score, 1) and leftest_digit(player_score, 2) < leftest_digit(opponent_score, 2)
    # END PROBLEM 4