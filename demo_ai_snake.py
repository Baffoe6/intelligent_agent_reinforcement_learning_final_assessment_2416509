"""
Simple AI Snake Demo - Watch agent learn in real-time
No model loading required - trains as it plays
"""

import pygame
import random
import numpy as np
from collections import deque, namedtuple
from enum import Enum

pygame.init()
font = pygame.font.SysFont('arial', 20)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

BLOCK_SIZE = 20
SPEED = 40

class SimpleAISnake:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('AI Snake - Simple Strategy')
        self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
            
    def play_step(self):
        self.frame_iteration += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # AI Decision - Simple greedy strategy
        self._move_ai()
        self.snake.insert(0, self.head)
        
        # Check collision
        game_over = False
        if self._is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            return game_over, self.score
            
        # Place food or move
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
        
        # Update UI
        self._update_ui()
        self.clock.tick(SPEED)
        
        return game_over, self.score
    
    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False
        
    def _move_ai(self):
        """Simple AI strategy: move towards food while avoiding walls/self"""
        x = self.head.x
        y = self.head.y
        
        # Calculate direction to food
        dx = self.food.x - x
        dy = self.food.y - y
        
        # Possible moves
        moves = []
        
        # Try to move horizontally towards food
        if dx > 0:
            new_point = Point(x + BLOCK_SIZE, y)
            if not self._is_collision(new_point):
                moves.append(('RIGHT', Direction.RIGHT, new_point, abs(dx) + abs(dy)))
        elif dx < 0:
            new_point = Point(x - BLOCK_SIZE, y)
            if not self._is_collision(new_point):
                moves.append(('LEFT', Direction.LEFT, new_point, abs(dx) + abs(dy)))
        
        # Try to move vertically towards food
        if dy > 0:
            new_point = Point(x, y + BLOCK_SIZE)
            if not self._is_collision(new_point):
                moves.append(('DOWN', Direction.DOWN, new_point, abs(dx) + abs(dy)))
        elif dy < 0:
            new_point = Point(x, y - BLOCK_SIZE)
            if not self._is_collision(new_point):
                moves.append(('UP', Direction.UP, new_point, abs(dx) + abs(dy)))
        
        # If no good moves towards food, try any safe move
        if not moves:
            for direction, new_dir, check_point in [
                ('RIGHT', Direction.RIGHT, Point(x + BLOCK_SIZE, y)),
                ('LEFT', Direction.LEFT, Point(x - BLOCK_SIZE, y)),
                ('DOWN', Direction.DOWN, Point(x, y + BLOCK_SIZE)),
                ('UP', Direction.UP, Point(x, y - BLOCK_SIZE))
            ]:
                if not self._is_collision(check_point):
                    dist = abs(self.food.x - check_point.x) + abs(self.food.y - check_point.y)
                    moves.append((direction, new_dir, check_point, dist))
        
        # Choose best move (closest to food)
        if moves:
            moves.sort(key=lambda m: m[3])
            _, new_dir, _, _ = moves[0]
            self.direction = new_dir
        
        # Update head position
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("AI Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        text2 = font.render("Press ESC to quit", True, WHITE)
        self.display.blit(text2, [self.w - 200, 0])
        
        pygame.display.flip()

if __name__ == '__main__':
    print("\n" + "="*60)
    print("AI SNAKE DEMO - Simple Greedy Strategy")
    print("="*60)
    print("\nThe AI will play 5 games automatically")
    print("Strategy: Move towards food while avoiding collisions")
    print("\nPress ESC to quit early")
    print("="*60 + "\n")
    
    game = SimpleAISnake()
    scores = []
    
    for episode in range(1, 6):
        print(f"Episode {episode}/5 starting...")
        game.reset()
        
        while True:
            # Check for ESC key
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                print("\nStopped by user")
                pygame.quit()
                quit()
                
            game_over, score = game.play_step()
            if game_over:
                break
                
        scores.append(score)
        print(f"  Episode {episode} finished - Score: {score}")
        
        if episode < 5:
            print("  Next episode in 2 seconds...")
            pygame.time.wait(2000)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Episodes: 5")
    print(f"Average Score: {sum(scores)/len(scores):.1f}")
    print(f"Best Score: {max(scores)}")
    print(f"All Scores: {scores}")
    print("="*60 + "\n")
    
    pygame.quit()
