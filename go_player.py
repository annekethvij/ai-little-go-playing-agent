import sys
import argparse
import copy
from copy import deepcopy
from random import choice, random
import random

import numpy as np


class GoGame:
    def __init__(self, n):
        self.size = n
        self.X_move = True
        self.died_pieces = []
        self.n_move = 0
        self.max_move = n * n - 1
        self.komi = n / 2
        self.verbose = False
        self.previous_board = None
        self.board = None

    def detect_line_stones(self, row, col, stone_color, horizontal=True):
        stones_in_line = [(row, col)]
        current_row, current_col = row, col
        if horizontal:
            directions = [(0, 1), (0, -1)]
        else:
            directions = [(1, 0), (-1, 0)]

        for dr, dc in directions:
            while True:
                current_row, current_col = current_row + dr, current_col + dc
                if (
                        0 <= current_row < self.size
                        and 0 <= current_col < self.size
                        and self.board[current_row][current_col] == stone_color
                ):
                    stones_in_line.append((current_row, current_col))
                else:
                    break

        return stones_in_line

    def calculate_line_liberty_score(self, line_group, go):
        liberties = go.find_group_liberties(go.board, line_group)
        if liberties is not None:
            num_stones = len(line_group)
            num_liberties = len(liberties)
            # Calculate a weighted score based on both stones and liberties
            return num_stones * num_liberties
        return 0

    def find_died_pieces(self, stone_color):
        died_pieces = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == stone_color:
                    if not self.find_liberty(i, j):
                        died_pieces.append((i, j))
        return died_pieces

    def init_board(self, n):
        self.board = [[0 for _ in range(n)] for _ in range(n)]

    def set_board(self, stone_color, previous_board, board):
        for i in range(self.size):
            for j in range(self.size):
                if previous_board[i][j] == stone_color and board[i][j] != stone_color:
                    self.died_pieces.append((i, j))
        self.previous_board = previous_board
        self.board = board

    def compare_board(self, board1, board2):
        for i in range(self.size):
            for j in range(self.size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def copy_board(self):
        return copy.deepcopy(self)

    def find_liberty(self, i, j):
        ally_members = self.ally_dfs(i, j)
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1])
            for piece in neighbors:
                if self.board[piece[0]][piece[1]] == 0:
                    return True
        return False

    def valid_place_check(self, i, j, stone_color):
        if not (i >= 0 and i < len(self.board)) or not (j >= 0 and j < len(self.board)):
            return False
        if self.board[i][j] != 0:
            return False

        test_go = self.copy_board()
        test_board = test_go.board
        test_board[i][j] = stone_color
        test_go.update_board(test_board)
        if test_go.find_liberty(i, j):
            return True

        test_go.remove_died_pieces(3 - stone_color)
        if not test_go.find_liberty(i, j):
            return False
        else:
            if self.died_pieces and self.compare_board(self.previous_board, test_go.board):
                return False
        return True

    def place_chess(self, i, j, stone_color):
        if not self.valid_place_check(i, j, stone_color):
            return False
        self.previous_board = copy.deepcopy(self.board)
        self.board[i][j] = stone_color
        self.update_board(self.board)
        return True

    def find_liberties(self, i, j, stone_color):
        current_board = self.board
        board_size = self.size

        # Make a copy of the current board to avoid modifying the original
        current_board_copy = [row[:] for row in current_board]

        # Place the stone at position (i, j)
        current_board_copy[i][j] = stone_color

        # Initialize an empty set to store liberties
        liberties = set()

        # Define the neighboring positions
        neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]

        for row, col in neighbors:
            if 0 <= row < board_size and 0 <= col < board_size:
                if current_board_copy[row][col] == 0:
                    # If the neighboring position is empty, it's a liberty
                    liberties.add((row, col))

        # Convert the set of liberties to a list
        liberties_list = list(liberties)

        return liberties_list

    def remove_died_pieces(self, stone_color):
        died_pieces = self.find_died_pieces(stone_color)
        if not died_pieces:
            return []
        self.remove_certain_pieces(died_pieces)
        return died_pieces

    def get_valid_moves(self, go, stone_color):
        valid_moves = []
        for i in range(go.size):
            for j in range(go.size):
                if board[i][j] == 0:
                    if go.valid_place_check(i, j, stone_color):
                        valid_moves.append((i, j))
        return valid_moves

    def calculate_mobility_score(self, stone_color):
        board = self.board

        mobility_score = 0
        board_size = len(board)

        legal_moves = self.get_valid_moves(go, stone_color)
        mobility_score = len(legal_moves)

        return mobility_score

    def remove_certain_pieces(self, positions):
        for piece in positions:
            self.board[piece[0]][piece[1]] = 0
        self.update_board(self.board)

    def detect_neighbor(self, i, j):
        neighbors = []
        if i > 0: neighbors.append((i - 1, j))
        if i < len(self.board) - 1: neighbors.append((i + 1, j))
        if j > 0: neighbors.append((i, j - 1))
        if j < len(self.board) - 1: neighbors.append((i, j + 1))
        return neighbors

    def detect_neighbor_ally(self, i, j):
        neighbors = self.detect_neighbor(i, j)
        group_allies = []
        for piece in neighbors:
            if self.board[piece[0]][piece[1]] == self.board[i][j]:
                group_allies.append(piece)
        return group_allies

    def ally_dfs(self, i, j):
        stack = [(i, j)]
        ally_members = []
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            neighbor_allies = self.detect_neighbor_ally(piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members

    def update_board(self, new_board):
        self.board = new_board

    def game_end(self, stone_color, action="MOVE"):
        if self.n_move >= self.max_move:
            return True
        if self.compare_board(self.previous_board, self.board) and action == "PASS":
            return True
        return False

    def score(self, piece_type):
        board = self.board
        cnt = 0
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == piece_type:
                    cnt += 1
        return cnt

    def at_edge(self, i, j):
        if i == 0 or i == self.size - 1 or j == 0 or j == self.size - 1:
            return True
        return False

    def at_corner(self, i, j):
        if i == 0 or i == self.size - 1:
            if j == 0 or j == self.size - 1:
                return True
        return False

    def detect_diagonal(self, i, j):
        diagonals = []
        if i > 0 and j > 0:
            diagonals.append((i - 1, j - 1))
        if i > 0 and j < self.size - 1:
            diagonals.append((i - 1, j + 1))
        if i < self.size - 1 and j > 0:
            diagonals.append((i + 1, j - 1))
        if i < self.size - 1 and j < self.size - 1:
            diagonals.append((i + 1, j + 1))
        return diagonals

    def count_self_group_liberties(self, stone_color):
        liberty_count = 0
        liberties = 0
        large_self_groups = go.find_large_groups(stone_color)
        if large_self_groups is not None:
            for group in large_self_groups:
                liberties = go.find_group_liberties(go.board, group)
                if liberties is not None:
                    liberty_count = len(liberties)
                    # print("liberty count: ", liberty_count)
        return liberty_count

    def count_adjacent_empty_points(self, x, y):
        empty_count = 0

        # Check all adjacent positions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 5 and 0 <= new_y < 5 and self.board[new_x][new_y] == 0:
                empty_count += 1

        return empty_count

    def calculate_positional_score(self, stone_color):
        score = 0
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] == stone_color:
                    score += self.count_adjacent_empty_points(x, y)
        return score

    def calculate_connectedness_score(self, stone_color):
        board = self.board
        visited = [[False for _ in range(5)] for _ in range(5)]

        group_count = 0

        for row in range(5):
            for col in range(5):
                if board[row][col] == stone_color and not visited[row][col]:
                    group_count += 1
                    self.dfs(row, col, stone_color, board, visited)

        return group_count

    def dfs(self, row, col, stone_color, board, visited):
        visited[row][col] = True
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < 5 and 0 <= c < 5 and board[r][c] == stone_color and not visited[r][c]:
                self.dfs(r, c, stone_color, board, visited)

    def calculate_balance_score(self, stone_color):
        board = self.board

        balance_score = 0
        board_size = len(board)

        # Number of stones in each quadrant
        top_left = sum(1 for row in range(3) for col in range(3) if board[row][col] == stone_color)
        top_right = sum(1 for row in range(3) for col in range(2, 5) if board[row][col] == stone_color)
        bottom_left = sum(1 for row in range(2, 5) for col in range(3) if board[row][col] == stone_color)
        bottom_right = sum(1 for row in range(2, 5) for col in range(2, 5) if board[row][col] == stone_color)

        balance_score += 0.2 * (abs(top_left - top_right) + abs(bottom_left - bottom_right))

        return balance_score

    def calculate_territory_score(self, stone_color):
        territory_score = 0
        board = self.board
        board_size = len(board)

        for row in range(board_size):
            for col in range(board_size):
                if board[row][col] == 0:  # Empty intersection
                    is_surrounded = True

                    # Check all adjacent intersections
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        r, c = row + dr, col + dc
                        if 0 <= r < board_size and 0 <= c < board_size:
                            if board[r][c] != stone_color:
                                is_surrounded = False
                                break

                    if is_surrounded:
                        territory_score += 1

        return territory_score

    def find_large_groups(self, stone_color):
        large_enemy_groups = set()
        visited = [[False for _ in range(self.size)] for _ in range(self.size)]

        for i in range(self.size):
            for j in range(self.size):
                if not visited[i][j] and self.board[i][j] == stone_color:
                    group = self.ally_dfs(i, j)
                    if len(group) >= 3:
                        large_enemy_groups.add(frozenset(group))

        return [list(group) for group in large_enemy_groups]

    def opponent_move(self):

        if np.array_equal(self.board, self.previous_board):
            return None
        for i in range(self.size):
            for j in range(self.size):
                if self.previous_board[i][j] != self.previous_board[i][j] and self.board[i][j] != 0:
                    return i, j

    def get_diagnol_score(self, stone_color):
        diagonal_score = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.valid_place_check(i, j, stone_color):
                    if i == j or i + j == self.size - 1:
                        diagonal_score += 100
        return diagonal_score

    def calculate_rhombus_reward(self, stone_color):
        rhombus_reward = 0

        # Define the positions for the four corners of the central 3x3 square
        central_corners = [(1, 1), (1, 3), (3, 1), (3, 3)]

        for corner in central_corners:
            x, y = corner
            # Check for a rhombus pattern starting from the current corner
            if (
                    self.is_valid_rhombus_pattern(x, y, stone_color)
                    and self.board[x][y] == stone_color
            ):
                # Reward for creating an ally stone rhombus pattern
                rhombus_reward += 200

        return rhombus_reward

    def is_valid_rhombus_pattern(self, x, y, stone_color):
        rhombus_pattern = [
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2),
            (2, 0), (2, 1), (2, 2)
        ]

        for dx, dy in rhombus_pattern:
            new_x, new_y = x + dx, y + dy
            # Check if the position is within the board boundaries
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                if self.board[new_x][new_y] != stone_color:
                    return False

        return True

    def calculate_influence_score(self, stone_color):
        influence_score = 0

        board = go.board
        board_size = go.size

        position_weights = [
            [2.5, 1.5, 1.5, 1.5, 2.5],
            [1.5, 2.0, 2.5, 2.5, 1.5],
            [1.5, 2.5, 2.0, 2.5, 1.5],
            [1.5, 2.5, 2.5, 2.0, 1.5],
            [2.5, 1.5, 1.5, 1.5, 2.5]
        ]

        for row in range(board_size):
            for col in range(board_size):
                if board[row][col] == stone_color:
                    influence_score += position_weights[row][col]
        # print("influence socre is: ", influence_score)
        return influence_score

    def check_l_shape(self, stone_color):
        corner_positions = [(0, 0), (0, go.size - 1), (go.size - 1, 0), (go.size - 1, go.size - 1)]

        l_shape_penalty = 0

        for corner in corner_positions:
            if go.board[corner[0]][corner[1]] == 0:
                diagonal_neighbors = go.detect_diagonal(corner[0], corner[1])

                for diagonal in diagonal_neighbors:
                    if go.board[diagonal[0]][diagonal[1]] == stone_color:
                        adjacent_corner_neighbors = go.detect_neighbor(corner[0], corner[1])
                        corner_ally = 0
                        for corner_neighbor in adjacent_corner_neighbors:
                            if go.board[corner_neighbor[0]][corner_neighbor[1]] == stone_color:
                                corner_ally += 1

                        if corner_ally == 1:
                            # Penalize for forming an L shape around the corner
                            l_shape_penalty -= 400

        return l_shape_penalty

    def get_capture_score(self, my_stone_type):
        capture_score = 0
        for i in range(5):
            for j in range(5):
                if go.board[i][j] == my_stone_type:
                    if i > 0 and go.board[i - 1][j] == (3 - my_stone_type):  # Check above
                        capture_score += 1
                    if i < 4 and go.board[i + 1][j] == (3 - my_stone_type):  # Check below
                        capture_score += 1
                    if j > 0 and go.board[i][j - 1] == (3 - my_stone_type):  # Check left
                        capture_score += 1
                    if j < 4 and go.board[i][j + 1] == (3 - my_stone_type):  # Check right
                        capture_score += 1
        return capture_score

    def find_group_liberties(self, board, group_stones, visited=None):
        if visited is None:
            visited = set()

        liberties = set()
        for stone in group_stones:
            visited.add(stone)
            x, y = stone

            neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            for neighbor in neighbors:
                if (
                        0 <= neighbor[0] < len(board) and
                        0 <= neighbor[1] < len(board[0]) and
                        board[neighbor[0]][neighbor[1]] == 0
                ):
                    liberties.add(neighbor)
                elif (
                        0 <= neighbor[0] < len(board) and
                        0 <= neighbor[1] < len(board[0]) and
                        board[neighbor[0]][neighbor[1]] == board[x][y] and
                        neighbor not in visited
                ):
                    liberties.update(self.find_group_liberties(board, [neighbor], visited))

        return liberties

    def count_captured_stones(self, stone_color):
        captured_stones = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    # Check if placing a stone at (i, j) would capture any enemy stones
                    if self.valid_place_check(i, j, stone_color):
                        # Make a copy of the board to simulate the move
                        test_board = deepcopy(self.board)
                        test_board[i][j] = stone_color

                        # Remove any captured enemy stones
                        captured_enemy_stones = self.remove_died_pieces(3 - stone_color)

                        # Count the number of captured enemy stones
                        captured_stones += len(captured_enemy_stones)

        return captured_stones


class AIPlayer:
    def __init__(self):
        pass

    def get_input(self, go, stone_color):
        valid_moves = go.get_valid_moves(go, stone_color)

        start_move = (2, 2)
        # sys.exit()
        if go.n_move == 0 and start_move in valid_moves:
            return start_move

        if not valid_moves:
            return "PASS"

        _, best_move = self.minimax_alpha_beta(go, stone_color)

        if best_move not in valid_moves:
            return "PASS"
        else:
            print("best_move: ", best_move)
            return best_move

    def evaluate_board(self, go, stone_color, calculate_positional_score=None):

        evaluation = 0

        current_board = go.board
        board_size = go.size
        captured_enemy_groups_score = 0
        ko_penalty = 0
        low_self_liberty_penalty = 0

        large_enemy_groups_penalty = 0
        large_self_groups_reward = 0
        player_stones, opponent_stones = 0, 0
        player_liberties, enemy_liberties = 0, 0

        #####################

        rhombus_reward = go.calculate_rhombus_reward(stone_color)
        capture_score = go.get_capture_score(stone_color)
        captured_stones = go.count_captured_stones(stone_color)

        l_shape_penalty = go.check_l_shape(3 - stone_color)

        balance_score = go.calculate_balance_score(stone_color)

        large_enemy_groups = go.find_large_groups(3 - stone_color)
        large_self_groups = go.find_large_groups(stone_color)
        # num_large_self_groups = len(large_self_groups)

        # Penalize for large enemy groups
        for group in large_enemy_groups:
            group_size = len(group)
            if group_size >= 4:
                large_enemy_groups_penalty -= 100  # Penalize for very large enemy groups
            elif group_size >= 3:
                large_enemy_groups_penalty -= 50  # Penalize for large enemy groups
            else:
                 large_enemy_groups_penalty -= 10  # Small penalty for smaller enemy groups

        # Reward for large self groups
        for group in large_self_groups:
            group_size = len(group)
            if group_size >= 4:
                large_self_groups_reward += 100
            elif group_size >= 3:
                large_self_groups_reward += 50
            else:
                large_self_groups_reward += 10

        # Logic handling number of self stones, enemy stone count and libarties

        player_stones = go.score(stone_color)
        opponent_stones = go.score(opponent_stones)
        player_liberties=0
        enemy_liberties=0

        low_enemy_liberty_reward=0
        # Logic for less liberty penalty
        self_group_liberties = go.count_self_group_liberties(stone_color)
        if self_group_liberties < 2:
            low_self_liberty_penalty -= 400


        enemy_group_liberties = go.count_self_group_liberties(3-stone_color)
        if enemy_group_liberties < 1:
            low_enemy_liberty_reward += 1000
        elif enemy_group_liberties < 2:
                low_enemy_liberty_reward += 400

        horizontal_line_score = 0
        vertical_line_score = 0
        enemy_vertical_line_score=0
        enemy_horizontal_line_score=0

        for row in range(go.size):
            for col in range(go.size):
                #The logic for liberties
                    liberties = go.find_liberties(row, col, stone_color)
                    if liberties is not None:
                        player_liberties = len(liberties)
                    else:
                        player_liberties = 0

                    e_liberties = go.find_liberties(row, col, 3 - stone_color)  # Use e_liberties for enemy liberties
                    if e_liberties is not None:
                        enemy_liberties = len(e_liberties)
                    else:
                        enemy_liberties = 0
                #The logic for x and y axis complete lines
                    if go.board[row][col] == stone_color:
                        # Check horizontal line to the right
                        horizontal_line_group = go.detect_line_stones(row, col, stone_color, horizontal=True)
                        horizontal_line_score += go.calculate_line_liberty_score(horizontal_line_group, go)

                        # Check vertical line downward
                        vertical_line_group = go.detect_line_stones(row, col, stone_color, horizontal=False)
                        vertical_line_score += go.calculate_line_liberty_score(vertical_line_group, go)
                    else:
                        # Check horizontal line to the right
                        enemy_horizontal_line_group = go.detect_line_stones(row, col, 3- stone_color, horizontal=True)
                        enemy_horizontal_line_score += go.calculate_line_liberty_score(enemy_horizontal_line_group, go)

                        # Check vertical line downward
                        enemy_vertical_line_group = go.detect_line_stones(row, col, 3- stone_color, horizontal=False)
                        enemy_vertical_line_score += go.calculate_line_liberty_score(enemy_vertical_line_group, go)

            ##################
            # # stone 1 is black
        if stone_color == 1:
            evaluation = (player_stones) + (player_liberties) + 200*captured_stones +10*l_shape_penalty\
                         + rhombus_reward - capture_score \
                         + 10*horizontal_line_score + 10*vertical_line_score \
                         + 100*large_enemy_groups_penalty \
                         + 1000*low_self_liberty_penalty\

            # # stone 1 is white
        elif stone_color == 2:
                evaluation = (player_stones) + (player_liberties) \
                             + rhombus_reward - capture_score \
                             + horizontal_line_score + vertical_line_score \
                             + large_enemy_groups_penalty \
                             + 10*low_self_liberty_penalty \

        return evaluation

    ###############

    def minimax_alpha_beta(self, go, stone_color, depth=2, alpha=float('-inf'), beta=float('inf'),
                           maximizing_player=True):
        if depth == 0 or go.game_end(stone_color):
            return self.evaluate_board(go, stone_color), None

        valid_moves = go.get_valid_moves(go, stone_color)
        random.shuffle(valid_moves)

        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for move in valid_moves:
                go_copy = go.copy_board()
                go_copy.place_chess(move[0], move[1], stone_color)
                eval_move, _ = self.minimax_alpha_beta(go_copy, 3 - stone_color, depth - 1, alpha, beta, False)
                if eval_move > max_eval:
                    max_eval = eval_move
                    best_move = move
                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in valid_moves:
                go_copy = go.copy_board()
                go_copy.place_chess(move[0], move[1], stone_color)
                eval_move, _ = self.minimax_alpha_beta(go_copy, 3 - stone_color, depth - 1, alpha, beta, True)
                if eval_move < min_eval:
                    min_eval = eval_move
                    best_move = move
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break
            return min_eval, best_move


def readInput(n, path="input.txt"):
    with open(path, 'r') as f:
        lines = f.readlines()

        stone_type = int(lines[0])

        previous_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n + 1]]
        board = [[int(x) for x in line.rstrip('\n')] for line in lines[n + 1: 2 * n + 1]]

        return stone_type, previous_board, board


def writeOutput(result, path="output.txt"):
    res = ""
    if result == "PASS":
        res = "PASS"
    else:
        res += str(result[0]) + ',' + str(result[1])

    with open(path, 'w') as f:
        f.write(res)


def writePass(path="output.txt"):
    with open(path, 'w') as f:
        f.write("PASS")


if __name__ == "__main__":
    N = 5
    stone_color, previous_board, board = readInput(N)
    go = GoGame(N)
    go.set_board(stone_color, previous_board, board)
    player = AIPlayer()
    action = player.get_input(go, stone_color)
    writeOutput(action)
