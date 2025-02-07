#####     Player     #####
import numpy as np
import random

class PLAYER():
    """Player"""
    def __init__(self, side:int, name=None, mode:str="Random"):
        self._Side:int = side # Balck=1, White=2
        self._Mode:str = mode # Mode: Random/Human/AI
        self._Name:str = "Player" if name is None else str(name)
        self._Wins:int = 0 # Num of wins
    
    def __str__(self):
        return f"{self._Name}: Side-{self._Side}, Mode-{self._Mode}, Wins-{self._Wins}"
    
    def ACT(self, board) -> tuple[int, int]:
        """Make an action
        :param board (GomokuBoard): current board
        :return action: (y,x)"""
        y, x = random.choice(np.argwhere(board.getBoard()==0))
        return y, x
    
    def AddWin(self) -> int:
        self._Wins += 1
        return self._Wins
    
    def StartNew(self):
        """Start a new game"""
        pass
    
    def IsHuman(self) -> bool:
        """Whether is HumanPlayer"""
        return self._Mode == "Human"

if __name__ == "__main__":
    from _GomokuBoard import GomokuBoard
    pl1, pl2 = PLAYER(1), PLAYER(2)
    print(pl1)
    print(pl2)
    board = GomokuBoard()
    for i in range(10):
        y, x = pl1.ACT(board.copy())
        board.placeStone(y, x)
    print(board)
    print(np.sum(board.getBoard()))