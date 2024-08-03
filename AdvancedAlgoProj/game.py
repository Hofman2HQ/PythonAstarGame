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
WIDTH, HEIGHT = 1000, 600
GRID_SIZE = 10
CELL_SIZE = min(WIDTH, HEIGHT) // GRID_SIZE

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
BACKGROUND_COLOR = (240, 240, 240)
BUTTON_COLOR = (100, 100, 100)
BUTTON_HOVER_COLOR = (150, 150, 150)

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maze Game")
font = pygame.font.Font(pygame.font.get_default_font(), 24)
menu_font = pygame.font.Font(pygame.font.get_default_font(), 48)
button_font = pygame.font.Font(pygame.font.get_default_font(), 36)

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
        self.logs = 3
        self.goal = goal
        self.path = []
        self.model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
        self.train_data = []
        self.train_model()

    def move(self):
        if len(self.path) > 1:
            self.x, self.y = self.path[1]
            self.path = self.path[1:]

    def draw(self):
        screen.blit(self.image, (self.x * CELL_SIZE, self.y * CELL_SIZE))

    def train_model(self):
        if len(self.train_data) > 100:
            X, y = zip(*self.train_data)
            self.model.fit(X, y)
        else:
            X = np.random.rand(1000, 7)  # [my_x, my_y, opponent_x, opponent_y, logs_left, distance_to_goal, opponent_distance_to_goal]
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

# Game state
def initialize_players():
    global player1, player2, logs
    player1 = Player(0, 0, player1_img, (GRID_SIZE - 1, GRID_SIZE - 1))
    player2 = Player(GRID_SIZE - 1, GRID_SIZE - 1, player2_img, (0, 0))
    logs = []

def draw_grid():
    for x in range(0, GRID_SIZE * CELL_SIZE, CELL_SIZE):
        pygame.draw.line(screen, DARK_GREY, (x, 0), (x, GRID_SIZE * CELL_SIZE))
    for y in range(0, GRID_SIZE * CELL_SIZE, CELL_SIZE):
        pygame.draw.line(screen, DARK_GREY, (0, y), (GRID_SIZE * CELL_SIZE, y))

def draw():
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

    pygame.display.flip()

def heuristic(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])

def get_neighbors(x, y, current_logs, can_hop):
    neighbors = []
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
            if not any(log.x <= new_x < log.x + (2 if log.horizontal else 1) and
                       log.y <= new_y < log.y + (2 if not log.horizontal else 1) for log in current_logs):
                neighbors.append((new_x, new_y))
            elif can_hop:
                hop_x, hop_y = new_x + dx, new_y + dy
                if 0 <= hop_x < GRID_SIZE and 0 <= hop_y < GRID_SIZE:
                    if not any(log.x <= hop_x < log.x + (2 if log.horizontal else 1) and
                               log.y <= hop_y < log.y + (2 if not log.horizontal else 1) for log in current_logs):
                        neighbors.append((hop_x, hop_y))
    return neighbors

def a_star(start, goal, current_logs, can_hop):
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

        for next in get_neighbors(*current, current_logs, can_hop):
            new_cost = cost_so_far[current] + 1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                frontier.put((priority, next))
                came_from[next] = current

    return None

def place_log(player, opponent):
    if player.logs > 0:
        best_log = None
        max_length = 0
        for _ in range(10):  # Try up to 10 times
            horizontal = random.choice([True, False])
            if horizontal:
                x = random.randint(0, GRID_SIZE - 2)
                y = random.randint(0, GRID_SIZE - 1)
            else:
                x = random.randint(0, GRID_SIZE - 1)
                y = random.randint(0, GRID_SIZE - 2)

            # Check if the new log overlaps with any existing log
            if not any((log.x <= x < log.x + (2 if log.horizontal else 1) and
                        log.y <= y < log.y + (2 if not log.horizontal else 1)) or
                       (x <= log.x < x + (2 if horizontal else 1) and
                        y <= log.y < y + (2 if not horizontal else 1)) for log in logs):
                temp_log = Log(x, y, horizontal)
                temp_logs = logs + [temp_log]
                path1 = a_star((player.x, player.y), player.goal, temp_logs, False)
                path2 = a_star((opponent.x, opponent.y), opponent.goal, temp_logs, False)
                if path1 and path2 and len(path2) > max_length:
                    best_log = temp_log
                    max_length = len(path2)

        if best_log:
            logs.append(best_log)
            player.logs -= 1
            return True
    return False

def player_turn(player, opponent, mode):
    action = player.decide_action(opponent, logs)
    
    if action == 'place_log':
        if not place_log(player, opponent):
            action = 'move'
    
    if action in ['move', 'hop']:
        can_hop = (action == 'hop')
        player.path = a_star((player.x, player.y), player.goal, logs, can_hop)
        if player.path and len(player.path) > 1:
            player.move()

def check_win_condition(player):
    return (player.x, player.y) == player.goal

def clear_logs():
    global logs
    logs = []

def play_game():
    turn = 0
    move_counter1 = 0
    move_counter2 = 0
    mode1 = 'place_log'
    mode2 = 'place_log'
    running = True
    winner = None
    max_turns = 1000  # Prevent infinite games

    while running and turn < max_turns:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

        if mode1 == 'place_log' and mode2 == 'place_log':
            clear_logs()  # Clear logs before new placement phase

        if mode1 == 'place_log' and player1.logs == 0:
            mode1 = 'move'
            player1.path = a_star((player1.x, player1.y), player1.goal, logs, False)

        if mode2 == 'place_log' and player2.logs == 0:
            mode2 = 'move'
            player2.path = a_star((player2.x, player2.y), player2.goal, logs, False)

        if turn % 2 == 0:
            player_turn(player1, player2, mode1)
            if check_win_condition(player1):
                winner = "Player 1"
                running = False
            if mode1 == 'move':
                move_counter1 += 1
                if move_counter1 >= 5:
                    mode1 = 'place_log'
                    player1.logs = 3
                    move_counter1 = 0
        else:
            player_turn(player2, player1, mode2)
            if check_win_condition(player2):
                winner = "Player 2"
                running = False
            if mode2 == 'move':
                move_counter2 += 1
                if move_counter2 >= 5:
                    mode2 = 'place_log'
                    player2.logs = 3
                    move_counter2 = 0

        # If both players have finished their movement phase, clear the logs
        if mode1 == 'place_log' and mode2 == 'place_log':
            clear_logs()

        turn += 1
        draw()
        pygame.time.Clock().tick(2)  # Slower speed for better visualization

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

def draw_button(text, rect):
    pygame.draw.rect(screen, BUTTON_COLOR, rect)
    pygame.draw.rect(screen, BLACK, rect, 2)
    text_surface = button_font.render(text, True, BLACK)
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)

def choose_player_images():
    global player1_img, player2_img
    chosen1 = False
    chosen2 = False

    while not (chosen1 and chosen2):
        screen.fill(BACKGROUND_COLOR)

        # Draw player 1 options
        for i, img in enumerate(player1_images):
            screen.blit(img, (WIDTH // 4 - CELL_SIZE // 2, HEIGHT // 4 + i * (CELL_SIZE + 10)))

        # Draw player 2 options
        for i, img in enumerate(player2_images):
            screen.blit(img, (WIDTH * 3 // 4 - CELL_SIZE // 2, HEIGHT // 4 + i * (CELL_SIZE + 10)))

        # Draw selection text
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

def draw_text(text, position, font, color=BLACK):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=position)
    screen.blit(text_surface, text_rect)

def main_menu():
    start_text = menu_font.render("Maze Game", True, BLACK)
    start_rect = pygame.Rect((WIDTH // 2 - 100, HEIGHT // 2 - 100, 200, 50))
    exit_rect = pygame.Rect((WIDTH // 2 - 100, HEIGHT // 2 + 50, 200, 50))
    choose_rect = pygame.Rect((WIDTH // 2 - 150, HEIGHT // 2 + 150, 300, 50))

    num_games = 1
    num_games_text = menu_font.render(f"Number of Games: {num_games}", True, BLACK)
    num_games_rect = pygame.Rect((WIDTH // 2 - 150, HEIGHT // 2, 300, 50))

    while True:
        screen.fill(BACKGROUND_COLOR)
        title_rect = start_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 150))
        screen.blit(start_text, title_rect)

        draw_button("Start Game", start_rect)
        draw_button("Choose Players", choose_rect)
        draw_button("Exit", exit_rect)

        pygame.draw.rect(screen, BUTTON_COLOR, num_games_rect)
        pygame.draw.rect(screen, BLACK, num_games_rect, 2)
        num_games_text = menu_font.render(f"Number of Games: {num_games}", True, BLACK)
        num_games_text_rect = num_games_text.get_rect(center=num_games_rect.center)
        screen.blit(num_games_text, num_games_text_rect)

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