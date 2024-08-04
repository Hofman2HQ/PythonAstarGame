import pygame
import sys
import json
from queue import PriorityQueue
import random
import numpy as np
from sklearn.neural_network import MLPRegressor

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = pygame.display.Info().current_w, pygame.display.Info().current_h
GRID_SIZE = 10
CELL_SIZE = min(WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE)  # This ensures the grid fits within the window

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BROWN = (139, 69, 19)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GREY = (169, 169, 169)
DARK_GREY = (105, 105, 105)
BACKGROUND_COLOR = (200, 225, 255)
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER_COLOR = (100, 149, 237)
TEXT_COLOR = (255, 255, 255)
ARROW_COLOR = (50, 50, 50)

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maze Game")
font = pygame.font.Font(pygame.font.get_default_font(), 24)
menu_font = pygame.font.Font(pygame.font.get_default_font(), 48)
button_font = pygame.font.Font(pygame.font.get_default_font(), 36)
arrow_font = pygame.font.Font(pygame.font.get_default_font(), 48)

# Load the door and player images
door_image = pygame.image.load('door.png')
player1_images = [
    pygame.image.load('animalplayer1.jpg'),
    pygame.image.load('animalplayer2.jpg'),
    pygame.image.load('animalplayer3.jpg'),
    pygame.image.load('animalplayer4.jpg')
]
player2_images = [
    pygame.image.load('animalplayer1.jpg'),
    pygame.image.load('animalplayer2.jpg'),
    pygame.image.load('animalplayer3.jpg'),
    pygame.image.load('animalplayer4.jpg')
]

# Scale the images to fit the grid cells
door_image = pygame.transform.scale(door_image, (CELL_SIZE, CELL_SIZE))
player1_images = [pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE)) for img in player1_images]
player2_images = [pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE)) for img in player2_images]

# Initialize the chosen player images
player1_img = player1_images[0]
player2_img = player2_images[1]


class Player:
    def __init__(self, x, y, image, goal):
        self.x = x
        self.y = y
        self.image = image
        self.logs = 5
        self.goal = goal
        self.path = []
        self.model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
        self.train_data = []
        self.train_model()
        self.logs_placed = 0
        self.moves_left = 5
        self.has_jumped = False

    def move(self):
        if len(self.path) > 1 and self.moves_left > 0:
            next_x, next_y = self.path[1]
            if abs(next_x - self.x) + abs(next_y - self.y) > 1:
                self.has_jumped = True
            self.x, self.y = next_x, next_y
            self.path = self.path[1:]
            self.moves_left -= 1

    def draw(self):
        screen.blit(self.image, (self.x * CELL_SIZE, self.y * CELL_SIZE))

    def train_model(self):
        if len(self.train_data) > 100:
            X, y = zip(*self.train_data)
            self.model.fit(X, y)
        else:
            X = np.random.rand(1000, 7)
            y = np.random.rand(1000)
            self.model.fit(X, y)

    def load_training_data(self, data):
        self.train_data.extend(data)
        self.train_model()

    def decide_action(self, opponent, logs):
        my_distance = abs(self.x - self.goal[0]) + abs(self.y - self.goal[1])
        opponent_distance = abs(opponent.x - opponent.goal[0]) + abs(opponent.y - opponent.goal[1])
        X = np.array([[self.x, self.y, opponent.x, opponent.y, self.logs, my_distance, opponent_distance]])

        prediction = self.model.predict(X)[0]

        if prediction > 0.7 and self.logs > 0:
            action = 'place_log'
        elif prediction > 0.3:
            action = 'hop'
        else:
            action = 'move'

        self.train_data.append((X[0], 1 if action == 'place_log' else (0.5 if action == 'hop' else 0)))

        return action


class Log:
    def __init__(self, x, y, horizontal):
        self.x = x
        self.y = y
        self.horizontal = horizontal

    def draw(self):
        if self.horizontal:
            pygame.draw.rect(screen, BROWN,
                             (self.x * CELL_SIZE, self.y * CELL_SIZE + CELL_SIZE // 4, CELL_SIZE * 2, CELL_SIZE // 2))
        else:
            pygame.draw.rect(screen, BROWN,
                             (self.x * CELL_SIZE + CELL_SIZE // 4, self.y * CELL_SIZE, CELL_SIZE // 2, CELL_SIZE * 2))


def initialize_players():
    global player1, player2, logs
    player1 = Player(0, 0, player1_img, (GRID_SIZE - 1, GRID_SIZE - 1))
    player2 = Player(GRID_SIZE - 1, GRID_SIZE - 1, player2_img, (0, 0))
    logs = []
    player1.logs_placed = 0
    player2.logs_placed = 0
    player1.moves_left = 5
    player2.moves_left = 5
    player1.has_jumped = False
    player2.has_jumped = False


def draw_grid():
    for x in range(0, GRID_SIZE * CELL_SIZE, CELL_SIZE):
        pygame.draw.line(screen, DARK_GREY, (x, 0), (x, GRID_SIZE * CELL_SIZE))
    for y in range(0, GRID_SIZE * CELL_SIZE, CELL_SIZE):
        pygame.draw.line(screen, DARK_GREY, (0, y), (GRID_SIZE * CELL_SIZE, y))


def draw(mode):
    screen.fill(BACKGROUND_COLOR)
    draw_grid()
    for log in logs:
        log.draw()

    screen.blit(door_image, (0 * CELL_SIZE, 0 * CELL_SIZE))
    screen.blit(door_image, ((GRID_SIZE - 1) * CELL_SIZE, (GRID_SIZE - 1) * CELL_SIZE))

    player1.draw()
    player2.draw()

    for player in [player1, player2]:
        if player.path:
            for i in range(len(player.path) - 1):
                start = player.path[i]
                end = player.path[i + 1]
                start_pos = (start[0] * CELL_SIZE + CELL_SIZE // 2, start[1] * CELL_SIZE + CELL_SIZE // 2)
                end_pos = (end[0] * CELL_SIZE + CELL_SIZE // 2, end[1] * CELL_SIZE + CELL_SIZE // 2)
                pygame.draw.line(screen, GREEN if player == player1 else YELLOW, start_pos, end_pos, 2)

    info_text = f"Player 1 Logs: {player1.logs}\nPlayer 2 Logs: {player2.logs}"
    info_surface = font.render(info_text, True, BLACK)
    screen.blit(info_surface, (GRID_SIZE * CELL_SIZE + 20, 20))

    phase_text = f"Current Phase: {'Log Placement' if mode == 'place_log' else 'Movement'}"
    phase_surface = font.render(phase_text, True, BLACK)
    screen.blit(phase_surface, (GRID_SIZE * CELL_SIZE + 20, 80))

    if mode == 'move':
        moves_text = f"Moves Left - Player 1: {player1.moves_left}, Player 2: {player2.moves_left}"
        moves_surface = font.render(moves_text, True, BLACK)
        screen.blit(moves_surface, (GRID_SIZE * CELL_SIZE + 20, 120))

    pygame.display.flip()


def heuristic(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])


def get_neighbors(x, y, current_logs, can_hop, has_jumped):
    neighbors = []
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
            if not any(log.x <= new_x < log.x + (2 if log.horizontal else 1) and
                       log.y <= new_y < log.y + (2 if not log.horizontal else 1) for log in current_logs):
                neighbors.append((new_x, new_y))
            elif can_hop and not has_jumped:
                hop_x, hop_y = new_x + dx, new_y + dy
                if 0 <= hop_x < GRID_SIZE and 0 <= hop_y < GRID_SIZE:
                    if not any(log.x <= hop_x < log.x + (2 if log.horizontal else 1) and
                               log.y <= hop_y < log.y + (2 if not log.horizontal else 1) for log in current_logs):
                        neighbors.append((hop_x, hop_y))
    return neighbors


def a_star(start, goal, current_logs, can_hop, has_jumped):
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not frontier.empty():
        current = frontier.get()[1]

        if current == goal:
            path = []
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return list(reversed(path))

        for next in get_neighbors(*current, current_logs, can_hop, has_jumped):
            new_cost = cost_so_far[current] + 1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                frontier.put((priority, next))
                came_from[next] = current

    return None


def logs_overlap(new_log, existing_logs):
    for log in existing_logs:
        if new_log.horizontal == log.horizontal:
            if new_log.horizontal:
                if new_log.y == log.y and (new_log.x < log.x + 2 and log.x < new_log.x + 2):
                    return True
            else:
                if new_log.x == log.x and (new_log.y < log.y + 2 and log.y < new_log.y + 2):
                    return True
        else:
            if new_log.horizontal:
                if (log.x <= new_log.x < log.x + 1 or log.x <= new_log.x + 1 < log.x + 1) and \
                   (new_log.y <= log.y < new_log.y + 1 or new_log.y <= log.y + 1 < new_log.y + 1):
                    return True
            else:
                if (new_log.x <= log.x < new_log.x + 1 or new_log.x <= log.x + 1 < new_log.x + 1) and \
                   (log.y <= new_log.y < log.y + 1 or log.y <= new_log.y + 1 < log.y + 1):
                    return True
    return False


def place_log(player, opponent):
    if player.logs > 0:
        for _ in range(50):
            horizontal = random.choice([True, False])
            if horizontal:
                x = random.randint(0, GRID_SIZE - 2)
                y = random.randint(0, GRID_SIZE - 1)
            else:
                x = random.randint(0, GRID_SIZE - 1)
                y = random.randint(0, GRID_SIZE - 2)

            new_log = Log(x, y, horizontal)

            if not logs_overlap(new_log, logs) and \
               not (x == player.x and y == player.y) and \
               not (x == opponent.x and y == opponent.y) and \
               not (x == 0 and y == 0) and \
               not (x == GRID_SIZE - 1 and y == GRID_SIZE - 1):

                temp_logs = logs + [new_log]
                path1 = a_star((player.x, player.y), player.goal, temp_logs, True, False)
                path2 = a_star((opponent.x, opponent.y), opponent.goal, temp_logs, True, False)

                if path1 and path2:
                    logs.append(new_log)
                    player.logs -= 1
                    return True

    return False


def player_turn(player, opponent, mode):
    if mode == 'place_log':
        if player.logs_placed < 5:
            if place_log(player, opponent):
                player.logs_placed += 1
    elif mode == 'move':
        if player.moves_left > 0:
            player.path = a_star((player.x, player.y), player.goal, logs, True, player.has_jumped)
            player.move()


def check_win_condition(player):
    return (player.x, player.y) == player.goal


def clear_logs():
    global logs
    logs = []


def play_game():
    turn = 0
    mode = 'place_log'
    running = True
    winner = None
    max_turns = 1000

    while running and turn < max_turns:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

        if mode == 'place_log' and player1.logs_placed == 5 and player2.logs_placed == 5:
            mode = 'move'
            player1.path = a_star((player1.x, player1.y), player1.goal, logs, True, player1.has_jumped)
            player2.path = a_star((player2.x, player2.y), player2.goal, logs, True, player2.has_jumped)
            player1.moves_left = 5
            player2.moves_left = 5
            player1.has_jumped = False
            player2.has_jumped = False

        if turn % 2 == 0:
            player_turn(player1, player2, mode)
            if mode == 'move' and check_win_condition(player1):
                winner = "Player 1"
                running = False
        else:
            player_turn(player2, player1, mode)
            if mode == 'move' and check_win_condition(player2):
                winner = "Player 2"
                running = False

        if mode == 'move' and (player1.moves_left == 0 and player2.moves_left == 0):
            mode = 'place_log'
            clear_logs()
            player1.logs = 5
            player2.logs = 5
            player1.logs_placed = 0
            player2.logs_placed = 0

        turn += 1
        draw(mode)
        pygame.time.Clock().tick(2)

    if winner:
        print(f"{winner} wins!")
    elif turn >= max_turns:
        print("Game ended in a draw (max turns reached)")
    else:
        print("Game ended without a winner")

    save_game_data(player1, player2, logs, winner)

    return winner


def save_game_data(player1, player2, logs, winner):
    game_data = {
        'player1': {'x': player1.x, 'y': player1.y, 'logs': player1.logs},
        'player2': {'x': player2.x, 'y': player2.y, 'logs': player2.logs},
        'logs': [{'x': log.x, 'y': log.y, 'horizontal': log.horizontal} for log in logs],
        'winner': winner,
        'training_data': {
            'player1': player1.train_data,
            'player2': player2.train_data
        }
    }
    try:
        with open('game_data.json', 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.append(game_data)

    with open('game_data.json', 'w') as file:
        json.dump(data, file, indent=4)


def load_training_data():
    try:
        with open('game_data.json', 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    player1_data = []
    player2_data = []

    for game in data:
        player1_data.extend(game['training_data']['player1'])
        player2_data.extend(game['training_data']['player2'])

    return player1_data, player2_data


def draw_button(text, rect, is_hovered=False):
    color = BUTTON_HOVER_COLOR if is_hovered else BUTTON_COLOR
    pygame.draw.rect(screen, color, rect, border_radius=10)
    pygame.draw.rect(screen, BLACK, rect, 2, border_radius=10)
    text_surface = button_font.render(text, True, TEXT_COLOR)
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)


def draw_text(text, position, font, color=BLACK):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=position)
    screen.blit(text_surface, text_rect)


def draw_arrows(num_games_rect):
    left_arrow = arrow_font.render("<", True, ARROW_COLOR)
    right_arrow = arrow_font.render(">", True, ARROW_COLOR)
    left_rect = left_arrow.get_rect(midright=(num_games_rect.left - 10, num_games_rect.centery))
    right_rect = right_arrow.get_rect(midleft=(num_games_rect.right + 10, num_games_rect.centery))
    screen.blit(left_arrow, left_rect)
    screen.blit(right_arrow, right_rect)
    return left_rect, right_rect


def choose_player_images():
    global player1_img, player2_img
    chosen1 = False
    chosen2 = False

    while not (chosen1 and chosen2):
        screen.fill(BACKGROUND_COLOR)

        for i, img in enumerate(player1_images):
            screen.blit(img, (WIDTH // 4 - CELL_SIZE // 2, HEIGHT // 4 + i * (CELL_SIZE + 10)))

        for i, img in enumerate(player2_images):
            screen.blit(img, (WIDTH * 3 // 4 - CELL_SIZE // 2, HEIGHT // 4 + i * (CELL_SIZE + 10)))

        draw_text("Choose Player 1", (WIDTH // 4, HEIGHT // 4 - 40), menu_font)
        draw_text("Choose Player 2", (WIDTH * 3 // 4, HEIGHT // 4 - 40), menu_font)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                for i, img in enumerate(player1_images):
                    if WIDTH // 4 - CELL_SIZE // 2 <= event.pos[0] <= WIDTH // 4 + CELL_SIZE // 2:
                        if HEIGHT // 4 + i * (CELL_SIZE + 10) <= event.pos[1] <= HEIGHT // 4 + i * (CELL_SIZE + 10) + CELL_SIZE:
                            player1_img = img
                            chosen1 = True
                for i, img in enumerate(player2_images):
                    if WIDTH * 3 // 4 - CELL_SIZE // 2 <= event.pos[0] <= WIDTH * 3 // 4 + CELL_SIZE // 2:
                        if HEIGHT // 4 + i * (CELL_SIZE + 10) <= event.pos[1] <= HEIGHT // 4 + i * (CELL_SIZE + 10) + CELL_SIZE:
                            player2_img = img
                            chosen2 = True


def main_menu():
    start_text = menu_font.render("Maze Game", True, BLACK)
    start_rect = pygame.Rect((WIDTH // 2 - 150, HEIGHT // 2 - 100, 300, 50))
    choose_rect = pygame.Rect((WIDTH // 2 - 150, HEIGHT // 2 + 50, 300, 50))
    exit_rect = pygame.Rect((WIDTH // 2 - 150, HEIGHT // 2 + 150, 300, 50))

    num_games = 1
    num_games_text = menu_font.render(f"Number of Games: {num_games}", True, BLACK)
    num_games_rect = num_games_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))

    while True:
        screen.fill(BACKGROUND_COLOR)
        title_rect = start_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 200))
        screen.blit(start_text, title_rect)

        mouse_pos = pygame.mouse.get_pos()
        is_start_hovered = start_rect.collidepoint(mouse_pos)
        is_choose_hovered = choose_rect.collidepoint(mouse_pos)
        is_exit_hovered = exit_rect.collidepoint(mouse_pos)

        draw_button("Start Game", start_rect, is_start_hovered)
        draw_button("Choose Players", choose_rect, is_choose_hovered)
        draw_button("Exit", exit_rect, is_exit_hovered)

        num_games_text = menu_font.render(f"Number of Games: {num_games}", True, BLACK)
        num_games_rect = num_games_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(num_games_text, num_games_rect)

        left_arrow_rect, right_arrow_rect = draw_arrows(num_games_rect)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'exit', 0
            if event.type == pygame.MOUSEBUTTONDOWN:
                if start_rect.collidepoint(event.pos):
                    return 'start', num_games
                elif choose_rect.collidepoint(event.pos):
                    choose_player_images()
                elif exit_rect.collidepoint(event.pos):
                    return 'exit', 0
                elif left_arrow_rect.collidepoint(event.pos):
                    num_games = max(1, num_games - 1)
                elif right_arrow_rect.collidepoint(event.pos):
                    num_games += 1
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    num_games += 1
                elif event.key == pygame.K_DOWN:
                    num_games = max(1, num_games - 1)
                num_games_text = menu_font.render(f"Number of Games: {num_games}", True, BLACK)
                num_games_rect = num_games_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))


def main():
    player1_data, player2_data = load_training_data()
    initialize_players()
    player1.load_training_data(player1_data)
    player2.load_training_data(player2_data)

    while True:
        action, num_games = main_menu()

        if action == 'exit':
            pygame.quit()
            sys.exit()

        player1_wins = 0
        player2_wins = 0

        for game in range(num_games):
            print(f"Starting Game {game + 1}")
            initialize_players()
            winner = play_game()
            if winner == "Player 1":
                player1_wins += 1
            elif winner == "Player 2":
                player2_wins += 1

            player1.train_model()
            player2.train_model()

        print(f"Final Score - Player 1: {player1_wins}, Player 2: {player2_wins}")


if __name__ == "__main__":
    main()
