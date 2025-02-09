#####     AI Player     #####
#####     AlphaDog     #####
import numpy as np
import time, random
from torch import multiprocessing
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

import _Player
from _GomokuBoard import *
from _MCTS import *

#####     Experience Replay     #####
class ReplyMemory():
    def __init__(self, capacity:int, device):
        self.capacity = capacity
        self.dv = device
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)
    
    def push(self, state, mctsProb, result):
        """Put data into buffer
        :param state: 当前棋盘状态。
        :param mcts_probs: MCTS生成的动作概率分布。
        :param result: 最终结果 (1=Win, -1=Lose, 0=Draw)。
        """
        self.memory.append((state, mctsProb, result))

    def sample(self, batchSize:int):
        """Sample a batch randomly"""
        batch = random.sample(self.memory, batchSize)
        states, mctsProbs, results = zip(*batch)
        states = torch.concat([s for s in states]).to(self.dv) # [1,6,sz,sz] -> [bh,6,sz,sz]
        mctsProbs = torch.stack([p for p in mctsProbs]).to(self.dv) # [sz*sz] -> [bh,sz*sz]
        results = torch.tensor(results, dtype=torch.float32).unsqueeze(1).to(self.dv) # num -> [bh,1]
        return states, mctsProbs, results

#####     Nural Network     #####
class ResBlock(nn.Module):
    """Residual Block (inChnl=outChnl)"""
    def __init__(self, chnls:int, knSize=5):
        super().__init__()
        padding = (knSize-1) // 2
        self.conv1 = nn.Conv2d(chnls, chnls, knSize, padding=padding) # 5*5大卷积核，接归一化
        self.bn = nn.BatchNorm2d(chnls) # Batch Norm
        self.conv2 = nn.Conv2d(chnls, chnls*2, 1) # 1*1卷积核，接ReLU
        self.conv3 = nn.Conv2d(chnls*2, chnls, 1) # 1*1卷积核

    def forward(self, x):
        # f(x) = h(x) + x
        h = self.bn(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.conv3(h)
        return h + x

class GomokuNet(nn.Module):
    """Gomoku neural network: State -> Policy, Value"""
    def __init__(self, size=16):
        super().__init__()
        self.size = size
        self.convIn = nn.Sequential(nn.Conv2d(8, 96, kernel_size=5, padding=2), nn.BatchNorm2d(96),
                                    nn.Conv2d(96, 96, kernel_size=1), nn.ReLU())
        self.res = nn.Sequential(ResBlock(96, 5),
                                 ResBlock(96, 5),
                                 ResBlock(96, 5),
                                 ResBlock(96, 5),
                                 ResBlock(96, 5)) # ResBlocks * 5
        
        self.conv2 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=9, padding=4), nn.BatchNorm2d(32),
                                   nn.Conv2d(32, 32, kernel_size=1), nn.ReLU())
        
        self.convP = nn.Sequential(nn.Conv2d(128, 2, kernel_size=1),
                                   nn.BatchNorm2d(2), nn.ReLU())
        self.fcP = nn.Linear(2*size*size, size*size)
        
        self.convV = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1),
                                   nn.BatchNorm2d(1), nn.ReLU())
        self.fcV = nn.Sequential(nn.Linear(1*size*size, 256), nn.ReLU(),
                                 nn.Linear(256, 1))

    def forward(self, x): # [bh, 8, sz, sz]
        x1 = self.convIn(x) # [bh, 96, sz, sz]
        x1 = self.res(x1) # [bh, 96, sz, sz]

        x2 = self.conv2(x[:,0:3,:,:]) # [bh, 32, sz, sz]

        x = torch.concat([x1, x2], dim=1) # [bh, 128, sz, sz]

        policy = self.convP(x) # [bh, 2, sz, sz]
        policy = policy.view(policy.size(0), -1) # [bh, 2*sz*sz]
        policy = F.softmax(self.fcP(policy), dim=1) # [bh, sz*sz]

        value = self.convV(x) # [bh, 1, sz, sz]
        value = value.view(value.size(0), -1) # [bh, 1*sz*sz]
        value = torch.tanh(self.fcV(value)) # [bh, 1]

        return policy, value

#####     AlphaDog     #####
class AlphaDog(_Player.PLAYER):
    def __init__(self, side=1, size=16, modelPath=None, device=None):
        super().__init__(side, name="AlphaDog", mode="AI")
        self.Size = size # board size
        self.dv = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Net = GomokuNet(size=16).to(self.dv)
        if modelPath: self.Net.load_state_dict(torch.load(modelPath))
        self.MCTS = MCTS(self.Net, num_simulations=400, device=self.dv)
        self.Memory = ReplyMemory(15000, self.dv)

        self.tem = 1.05
        self.noise = True
        self.eps = 0.2
    
    def StartNew(self):
        self.MCTS.reset()
        
    def evalState(self, state:torch.Tensor):
        """State -> Policy, Value"""
        policy, value = self.Net.forward(state)
        return policy, value
    
    def selectAction(self, probs:np.ndarray, moveCount:int=0) -> int:
        """Select an action based on action probs (with Tau and Dir-noise)"""
        ### Temperature
        if moveCount <= 20: pass
        else:
            temperature = self.tem ** (moveCount-20)
            if temperature < 50: # (t=1.05 -> step=80)
                probs = probs ** temperature
                probs = probs / np.sum(probs)
            else:
                pos = np.argmax(probs)
                probs = np.zeros_like(probs)
                probs[pos] = 1.0
        ### Dirichlet Noise
        if self.noise:
            probs = (1-self.eps) * probs + self.eps * np.random.dirichlet([0.1]*len(probs))
        action = np.random.choice(len(probs), p=probs)
        return action
    
    def TakeAction(self, board:GomokuBoard) -> tuple:
        """Take an action: Board -> (r,c)"""
        actionProbs = self.MCTS.search(board)
        pos = self.selectAction(actionProbs, moveCount=len(board.moves))
        return pos // self.Size, pos % self.Size
    
    def playMode(self) -> 'AlphaDog':
        self.Net.eval()
        self.noise = False
        return self
    
    def ACT(self, board:GomokuBoard) -> tuple: ### Pygame 接口
        if True: # use MCTS search
            return self.TakeAction(board)
        else: # no search, only P-V net
            state = board.getStateAsT().to(self.dv)
            policy, _ = self.evalState(state)
            pos = torch.multinomial(policy.squeeze(), 1).item()
            return pos // self.Size, pos % self.Size
    
    def _SelfPlay(self) -> list:
        """Create self-play data
        :return: [(state, mctsProb, result), ...]
        """
        board = GomokuBoard()
        board.dv = self.dv
        self.StartNew()
        states, mctsProbs = [], []
        # Self-Play A Game
        with torch.no_grad():
            while not board.GameEnd:
                boardState = board.getStateAsT().to(self.dv)
                actionProbs = self.MCTS.search(board) # [sz*sz]
                pos = self.selectAction(actionProbs, moveCount=len(board.moves))
                if board.placeStone(pos // self.Size, pos % self.Size):
                    states.append(boardState)
                    mctsProbs.append(torch.from_numpy(actionProbs).float().to(self.dv))
        # Update Replay Memory (one for each move)
        result = 1 if board.Winner == 1 else -1 if board.Winner==2 else 0
        data = []
        for state, mctsProb in zip(states, mctsProbs): # old first
            data.append((state, mctsProb, result))
            result = -result
        return data
    
    def _CreateData(self, numPara:int, numGames:int):
        """Create self-play data (multi-process)
        :param numPara (int): number of parallel processes
        :param numGames (int): number of self-play games
        """
        modelDict = self.Net.state_dict()
        args = [(modelDict, torch.device("cuda" if torch.cuda.is_available() else "cpu")) for _ in range(numGames)]
        
        with multiprocessing.Pool(numPara) as pool: ### Parallel Self-Play
            allData = pool.starmap(_selfplayWorker, args)
        
        for data in allData: ### Update Replay Memory
            for state, mcts_prob, result in data:
                self.Memory.push(state.to(self.dv), mcts_prob.to(self.dv), result)
        torch.cuda.empty_cache()

    def TRAIN(self, lr=1e-3, batch_size=256):
        """Train the model
        :param lr (float): init learning rate
        :param batch_size (int): batch size for training
        """
        num_epochs = 40
        self.Net.train()
        print(f"Learning Rate: {lr},  Batch Size: {batch_size}")
        optimizer = torch.optim.AdamW(self.Net.parameters(), lr=lr, weight_decay=1e-4)
        
        self._CreateData(8, 16) # Warmup
        for epoch in range(1, num_epochs+1):
            ### Self-Play
            t_start = time.time()
            self._CreateData(8, 32)
            ### Sample and Step
            if not (len(self.Memory) > batch_size): continue
            else: print(len(self.Memory))
            iters = 0 # 迭代次数
            counter = 0 # 无效迭代计数器
            bestLoss = 100
            lossSum = 0.0
            while counter <= 3:
                optimizer.zero_grad()
                states_batch, mctsProbs_batch, rewards_batch = self.Memory.sample(batch_size)
                policy, value = self.evalState(states_batch)
                # LOSS: Policy-交叉熵损失 + Value-均方差损失 (+ WeightDecay)
                policyLoss = F.cross_entropy(policy.to(self.dv), mctsProbs_batch)
                valueLoss = F.mse_loss(value.to(self.dv), rewards_batch)
                loss = policyLoss + valueLoss
                # Backward and Update
                loss.backward()
                nn.utils.clip_grad_norm_(self.Net.parameters(), max_norm=1.0) # 梯度裁剪
                optimizer.step()
                # Check loss
                iters += 1
                lossSum += loss.item()
                if loss.item() >= bestLoss: counter += 1
                else: bestLoss, counter = loss.item(), 0
                torch.cuda.empty_cache() #
            print(f"Epoch-{epoch},  Loss:{lossSum/iters:.4f}~{bestLoss:.4f},  Iters:{iters},   Time:{(time.time()-t_start)/60:.1f}m")
            if epoch % 5 == 0:
                torch.save(self.Net.state_dict(), rootPath+f"model_{epoch}.pth")
            print()

def _selfplayWorker(modelDict, device:torch.device):
    """Self-Play Worker (for parallel processing)"""
    torch.cuda.set_per_process_memory_fraction(0.1) ###
    newPl = AlphaDog(device=device) # new player, same net
    newPl.Net.load_state_dict(modelDict)
    newPl.MCTS = MCTS(newPl.Net, num_simulations=400, device=device)
    return newPl._SelfPlay()

###
if __name__ == "__main__":
    rootPath = "/lustre/home/2100013210/MachineLearning/wzq/" ###
    print(f"CPUs: {multiprocessing.cpu_count()},  GPUs: {torch.cuda.device_count()}")
    multiprocessing.set_start_method('spawn') ###
    player = AlphaDog()
    print(f"{player},  {player.dv}")

    player.TRAIN()

    print("Completed")