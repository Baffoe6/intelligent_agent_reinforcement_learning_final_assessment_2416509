"""
Optimized Snake Game Environment
Enhanced environment with performance optimizations and configurable features
"""

import pygame
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum
from collections import namedtuple
import random

from ..utils.logging import get_logger


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')


class Colors:
    """Color constants"""
    WHITE = (255, 255, 255)
    RED = (200, 0, 0)
    BLUE1 = (0, 0, 255)
    BLUE2 = (0, 100, 255)
    BLACK = (0, 0, 0)
    GRAY = (128, 128, 128)
    GREEN = (0, 255, 0)


class SnakeEnvironment:
    """Optimized Snake game environment"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger()
        
        # Environment parameters
        self.width = config.get('width', 640)
        self.height = config.get('height', 480)
        self.block_size = config.get('block_size', 20)
        self.speed = config.get('speed', 40)
        self.has_walls = config.get('has_walls', False)
        
        # Calculate grid dimensions
        self.grid_width = self.width // self.block_size
        self.grid_height = self.height // self.block_size
        
        # Wall configuration
        self.walls = []
        if self.has_walls:
            self._setup_walls(config.get('wall_config', {}))
        
        # Pygame setup
        self.headless = config.get('headless', False)
        if not self.headless:
            pygame.init()
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Snake RL - Optimized')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('arial', 25)
        
        # Game state
        self.reset()
    
    def _setup_walls(self, wall_config: Dict[str, Any]):
        """Setup wall obstacles"""
        self.walls = []
        
        # Horizontal walls
        for wall_spec in wall_config.get('horizontal_walls', []):
            x, y, width, height = wall_spec
            for i in range(0, width, self.block_size):
                wall_point = Point(x + i, y)
                if self._is_valid_point(wall_point):
                    self.walls.append(wall_point)
        
        # Vertical walls
        for wall_spec in wall_config.get('vertical_walls', []):
            x, y, width, height = wall_spec
            for i in range(0, height, self.block_size):
                wall_point = Point(x, y + i)
                if self._is_valid_point(wall_point):
                    self.walls.append(wall_point)
        
        self.logger.info(f"Created {len(self.walls)} wall blocks")
    
    def _is_valid_point(self, point: Point) -> bool:
        """Check if point is within bounds"""
        return (0 <= point.x < self.width and 
                0 <= point.y < self.height and
                point.x % self.block_size == 0 and
                point.y % self.block_size == 0)
    
    def reset(self):
        """Reset environment to initial state"""
        # Initialize snake in center
        center_x = (self.grid_width // 2) * self.block_size
        center_y = (self.grid_height // 2) * self.block_size
        
        self.direction = Direction.RIGHT
        self.head = Point(center_x, center_y)
        self.snake = [
            self.head,
            Point(self.head.x - self.block_size, self.head.y),
            Point(self.head.x - 2 * self.block_size, self.head.y)
        ]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
        # Performance tracking
        self.max_frames = 100 * len(self.snake)  # Dynamic frame limit
    
    def _place_food(self):
        """Place food in valid location"""
        max_attempts = 1000
        attempts = 0
        
        while attempts < max_attempts:
            # Generate random grid position
            grid_x = random.randint(0, self.grid_width - 1)
            grid_y = random.randint(0, self.grid_height - 1)
            
            # Convert to pixel coordinates
            x = grid_x * self.block_size
            y = grid_y * self.block_size
            food_point = Point(x, y)
            
            # Check if position is valid
            if (food_point not in self.snake and 
                food_point not in self.walls):
                self.food = food_point
                return
            
            attempts += 1
        
        # Fallback: place food at first available position
        for x in range(0, self.width, self.block_size):
            for y in range(0, self.height, self.block_size):
                food_point = Point(x, y)
                if (food_point not in self.snake and 
                    food_point not in self.walls):
                    self.food = food_point
                    return
        
        # If no valid position found, use current head position (shouldn't happen)
        self.food = self.head
        self.logger.warning("Could not place food in valid position")
    
    def step(self, action: List[int]) -> Tuple[float, bool, int]:
        """Execute one environment step"""
        self.frame_iteration += 1
        
        # Handle pygame events (if not headless)
        if not self.headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        
        # Move snake
        self._move(action)
        self.snake.insert(0, self.head)
        
        # Check game over conditions
        reward = 0
        game_over = False
        
        # Collision check
        if self.is_collision():
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # Frame limit check (prevents infinite games)
        if self.frame_iteration > self.max_frames:
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # Food collision
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            # Update frame limit as snake grows
            self.max_frames = 100 * len(self.snake)
        else:
            self.snake.pop()
        
        # Small positive reward for staying alive
        reward += 0.01
        
        # Update display (if not headless)
        if not self.headless:
            self._update_display()
            self.clock.tick(self.speed)
        
        return reward, game_over, self.score
    
    def _move(self, action: List[int]):
        """Move snake based on action"""
        # Convert action to direction
        action_idx = np.argmax(action)
        
        # Direction mapping (clockwise)
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        current_idx = directions.index(self.direction)
        
        if action_idx == 0:  # Straight
            new_direction = self.direction
        elif action_idx == 1:  # Right turn
            new_direction = directions[(current_idx + 1) % 4]
        else:  # Left turn (action_idx == 2)
            new_direction = directions[(current_idx - 1) % 4]
        
        self.direction = new_direction
        
        # Calculate new head position
        x, y = self.head.x, self.head.y
        
        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size
        
        self.head = Point(x, y)
    
    def is_collision(self, point: Optional[Point] = None) -> bool:
        """Check collision at given point (or current head)"""
        if point is None:
            point = self.head
        
        # Boundary collision
        if (point.x >= self.width or point.x < 0 or 
            point.y >= self.height or point.y < 0):
            return True
        
        # Self collision
        if point in self.snake[1:]:
            return True
        
        # Wall collision
        if self.has_walls and point in self.walls:
            return True
        
        return False
    
    def is_collision_point(self, point: Point) -> bool:
        """Public method for collision checking (used by agent)"""
        return self.is_collision(point)
    
    def _update_display(self):
        """Update game display"""
        if self.headless:
            return
        
        self.display.fill(Colors.BLACK)
        
        # Draw snake
        for segment in self.snake:
            pygame.draw.rect(self.display, Colors.BLUE1, 
                           pygame.Rect(segment.x, segment.y, self.block_size, self.block_size))
            pygame.draw.rect(self.display, Colors.BLUE2,
                           pygame.Rect(segment.x + 4, segment.y + 4, 
                                     self.block_size - 8, self.block_size - 8))
        
        # Draw food
        pygame.draw.rect(self.display, Colors.RED,
                        pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size))
        
        # Draw walls
        if self.has_walls:
            for wall in self.walls:
                pygame.draw.rect(self.display, Colors.GRAY,
                               pygame.Rect(wall.x, wall.y, self.block_size, self.block_size))
        
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, Colors.WHITE)
        self.display.blit(score_text, [0, 0])
        
        # Draw walls indicator
        if self.has_walls:
            walls_text = self.font.render("WALLS ENABLED", True, Colors.WHITE)
            self.display.blit(walls_text, [0, 30])
        
        pygame.display.flip()
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information"""
        return {
            'snake_length': len(self.snake),
            'score': self.score,
            'frame': self.frame_iteration,
            'max_frames': self.max_frames,
            'head_position': self.head,
            'food_position': self.food,
            'direction': self.direction,
            'has_walls': self.has_walls
        }
    
    def close(self):
        """Clean up resources"""
        if not self.headless:
            try:
                pygame.quit()
            except:
                pass