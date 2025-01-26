import pygame
import numpy as np
from enum import IntEnum
from collections import deque

class Direction(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

class SnakeGame:
    def __init__(self, width=10, height=10, block_size=40):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.visited = set()
        self.reset()
        
        pygame.init()
        self.window = pygame.display.set_mode(
            (width * block_size, height * block_size)
        )
        self.clock = pygame.time.Clock()
        
    def reset(self):
        self.direction = Direction.RIGHT
        self.head = (self.width//2, self.height//2)
        self.snake = deque([self.head])
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.visited.clear()
        self.visited.add(self.head)
        return self._get_state()


    def _place_food(self):

        while True:
            self.food = (
                np.random.randint(0, self.width),
                np.random.randint(0, self.height)
            )
            if self.food not in self.snake:
                break

    def _get_front_cell(self):
        dx, dy = [(1,0), (0,1), (-1,0), (0,-1)][self.direction]
        return (self.head[0] + dx, self.head[1] + dy)

    def _get_left_cell(self):
        left_dir = (self.direction + 1) % 4
        dx, dy = [(1,0), (0,1), (-1,0), (0,-1)][left_dir]
        return (self.head[0] + dx, self.head[1] + dy)

    def _get_right_cell(self):
        right_dir = (self.direction - 1) % 4
        dx, dy = [(1,0), (0,1), (-1,0), (0,-1)][right_dir]
        return (self.head[0] + dx, self.head[1] + dy)

    def _relative_food_direction(self):
        dx = self.food[0] - self.head[0]
        dy = self.food[1] - self.head[1]
        
        if self.direction == Direction.RIGHT:
            rel_x, rel_y = dx, dy
        elif self.direction == Direction.UP:
            rel_x, rel_y = -dy, dx
        elif self.direction == Direction.LEFT:
            rel_x, rel_y = -dx, -dy
        else:  # DOWN
            rel_x, rel_y = dy, -dx
            
        return [
            rel_x > 0,
            rel_y > 0
        ]

    def _is_body_near(self, direction):
        if direction == "front":
            dx, dy = [(1,0), (0,1), (-1,0), (0,-1)][self.direction]
        elif direction == "left":
            dx, dy = [(1,0), (0,1), (-1,0), (0,-1)][(self.direction + 1) % 4]
        elif direction == "right":
            dx, dy = [(1,0), (0,1), (-1,0), (0,-1)][(self.direction - 1) % 4]
        else:
            return False
        
        for step in [1, 2]:
            check_pos = (self.head[0] + dx*step, self.head[1] + dy*step)
            if check_pos in list(self.snake)[1:]:
                return True
        return False

    def _get_state(self):
        danger = [
            self._is_collision(self._get_front_cell()),
            self._is_collision(self._get_left_cell()),
            self._is_collision(self._get_right_cell())
        ]
        food_dir = self._relative_food_direction()
        body_near = [
            self._is_body_near("front"),
            self._is_body_near("left"),
            self._is_body_near("right")
        ]
        return np.array(danger + food_dir + body_near, dtype=int)

    def _is_collision(self, pt=None):
        pt = pt or self.head
        return (
            pt[0] < 0 or pt[0] >= self.width or
            pt[1] < 0 or pt[1] >= self.height or
            pt in list(self.snake)[1:]
        )

    def step(self, action):
        self.frame_iteration += 1
        prev_head = self.head
        prev_food = self.food
        

        if action == 1:  # Right turn
            self.direction = (self.direction - 1) % 4
        elif action == 2:  # Left turn
            self.direction = (self.direction + 1) % 4
        
        # Move snake
        dx, dy = [(1,0), (0,1), (-1,0), (0,-1)][self.direction]
        self.head = (prev_head[0] + dx, prev_head[1] + dy)
        self.snake.appendleft(self.head)
        
        # Check game over
        reward = 0
        done = False
        if self._is_collision() or self.frame_iteration > 100*len(self.snake):
            reward = -20
            done = True
            return self._get_state(), reward, done
        
        # Check food
        if self.head == prev_food:
            self.score += 1
            self._place_food()
            reward = 20
        else:
            self.snake.pop()
            dist_before = abs(prev_food[0] - prev_head[0]) + abs(prev_food[1] - prev_head[1])
            dist_after = abs(prev_food[0] - self.head[0]) + abs(prev_food[1] - self.head[1])
            reward = 1 if dist_after < dist_before else -1
            
        return self._get_state(), reward, done

    def render(self, speed=20):
        """Render game with PyGame"""
        self.window.fill((0, 0, 0))
        
        # Draw snake
        for idx, pt in enumerate(self.snake):
            color = (0, 255, 0) if idx == 0 else (0, 200, 0)
            pygame.draw.rect(self.window, color, 
                pygame.Rect(
                    pt[0] * self.block_size,
                    pt[1] * self.block_size,
                    self.block_size - 1,
                    self.block_size - 1
                ))
        
        # Draw food
        pygame.draw.rect(self.window, (255, 0, 0), 
            pygame.Rect(
                self.food[0] * self.block_size,
                self.food[1] * self.block_size,
                self.block_size - 1,
                self.block_size - 1
            ))
        
        pygame.display.flip()
        self.clock.tick(speed)