#####     Player     #####
import pygame
from _Player import *

class HumanPl(PLAYER):
    """Human Player"""
    def __init__(self, side, name="机智如我"):
        super().__init__(side, name, mode="Human")
    
    def ACT(self, events:list, step) -> tuple[int, int] | None:
        """:param step: size of board grid"""
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_focused():
                x_pos, y_pos = pygame.mouse.get_pos()
                X, Y = (x_pos/step)-1 , (y_pos/step)-1
                X, Y = round(X), round(Y)
                return Y, X
        return None