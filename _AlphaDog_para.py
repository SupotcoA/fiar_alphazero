#####     AI Player     #####
#####     AlphaDog     #####
import numpy as np
import time, random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import multiprocessing
multiprocessing.set_sharing_strategy('file_system') ###

import _Player
from _GomokuBoard import *
from _MCTS import *

#####     Dataset     #####
class Dataset():
    """Data buffer for training"""
    def __init__(self, device, capacity:int=100000):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity
        self.dv = device

    def __len__(self):
        return len(self.memory)

    def clear(self):
        """Clear the data buffer"""
        self.memory = deque(maxlen=self.capacity)
    
    def push(self, state:torch.Tensor, mctsProb:torch.Tensor, result):
        """Put data into dataset (data augmentation: rotation & flip)"""
        size = state.shape[-1]
        state = state.squeeze(0) # [1, C, sz, sz] -> [C, sz, sz]
        mctsProb = mctsProb.reshape(size, size) # [sz*sz] -> [sz, sz] -> [sz*sz]
        for k in range(4): # Rotation * 4
            S_ = torch.rot90(state, k, dims=(-2,-1))
            P_ = torch.rot90(mctsProb, k, dims=(-2,-1)).reshape(-1)
            self.memory.append((S_, P_, result))
        state, mctsProb = torch.flip(state, dims=(-1,)), torch.flip(mctsProb, dims=(-1,))
        for k in range(4): # Rotation * 4
            S_ = torch.rot90(state, k, dims=(-2,-1))
            P_ = torch.rot90(mctsProb, k, dims=(-2,-1)).reshape(-1)
            self.memory.append((S_, P_, result))

    def sample(self, batchSize:int):
        """Sample a batch randomly"""
        batch = random.sample(self.memory, batchSize)
        states, mctsProbs, results = zip(*batch)
        states = torch.stack([s for s in states]).to(self.dv) # [C, sz, sz] -> [B, C, sz, sz]
        mctsProbs = torch.stack([p for p in mctsProbs]).to(self.dv) # [sz*sz] -> [B, sz*sz]
        results = torch.tensor(results, dtype=torch.float32).unsqueeze(1).to(self.dv) # num -> [B, 1]
        return states, mctsProbs, results

#####     Nural Network     #####
class ResBlock(nn.Module):
    """Residual Block (inChnl=outChnl)"""
    def __init__(self, chnls:int, knSize:int=5):
        super().__init__()
        padding = (knSize-1) // 2
        self.conv1 = nn.Conv2d(chnls, chnls, knSize, padding=padding) # 5*5大卷积核，接归一化
        self.bn = nn.BatchNorm2d(chnls) # Batch Norm
        self.conv2 = nn.Conv2d(chnls, chnls*2, 1) # 1*1卷积核，接ReLU
        self.conv3 = nn.Conv2d(chnls*2, chnls, 1) # 1*1卷积核

    def forward(self, x): # f(x) = h(x) + x
        h = self.bn(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.conv3(h)
        return h + x

class GomokuNet(nn.Module):
    """Gomoku neural network: State -> Policy, Value"""
    def __init__(self, size=16):
        super().__init__()
        self.size = size
        self.convIn = nn.Sequential(nn.Conv2d(5, 64, kernel_size=9, padding=4), nn.BatchNorm2d(64),
                                    nn.Conv2d(64, 128, kernel_size=1), nn.ReLU())
        self.res = nn.Sequential(ResBlock(128, 5),
                                 ResBlock(128, 5),
                                 ResBlock(128, 5)) # ResBlocks * 3
        
        self.convP = nn.Sequential(nn.Conv2d(128, 2, kernel_size=1),
                                   nn.BatchNorm2d(2), nn.ReLU())
        self.fcP = nn.Linear(2*size*size, size*size)
        
        self.convV = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1),
                                   nn.BatchNorm2d(1), nn.ReLU())
        self.fcV = nn.Sequential(nn.Linear(size*size, 128), nn.ReLU(),
                                 nn.Linear(128, 1))

    def forward(self, x): # [B, 5, sz, sz]
        x = self.convIn(x) # [B, 128, sz, sz]
        x = self.res(x) # [B, 128, sz, sz]

        policy = self.convP(x) # [B, 2, sz, sz]
        policy = policy.view(policy.size(0), -1) # [B, 2*sz*sz]
        policy = F.log_softmax(self.fcP(policy), dim=1) # [B, sz*sz]

        value = self.convV(x) # [B, 1, sz, sz]
        value = value.view(value.size(0), -1) # [B, 1*sz*sz]
        value = F.tanh(self.fcV(value)) # [B, 1]

        return policy, value

#####     AlphaDog     #####
class AlphaDog(_Player.PLAYER):
    """AlphaDog: A Gomoku AI Imitating AlphaZero"""
    def __init__(self, side=1, size=16, modelPath=None, device=None):
        super().__init__(side, name="AlphaDog", mode="AI")
        self.Size = size # board size
        self.dv = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Net = GomokuNet(size=16).to(self.dv)
        if modelPath: self.Net.load_state_dict(torch.load(modelPath, weights_only=True))
        self.MCTS = MCTS(self.Net, num_simulations=400, device=self.dv)
        self.Memory = Dataset(self.dv, capacity=80000)

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
        """Select an action based on probability (with Tau and Dir-noise)"""
        ### Temperature
        addTem = 20 # start to add temperature
        if moveCount <= addTem: pass
        else:
            temperature = self.tem ** (moveCount-addTem)
            if temperature <= 50: # t=1.05 -> step=80
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
    
    def ACT(self, board:GomokuBoard, useMCTS:bool=True) -> tuple:
        ### Pygame 接口
        if useMCTS: # use MCTS search
            return self.TakeAction(board)
        else: # no search, only P-V net
            policy, _ = self.evalState(board.getStateAsT().to(self.dv))
            policy = torch.exp(policy.squeeze())
            pos = torch.multinomial(policy, 1).item()
            return pos // self.Size, pos % self.Size
    
    def _SelfPlay(self) -> list[tuple]:
        """Create self-play data
        :return: [(state, mctsProb, result), ...]
        """
        board = GomokuBoard()
        board.dv = self.dv
        self.StartNew()
        # Self-Play A Game
        states, mctsProbs = [], []
        with torch.no_grad():
            while not board.GameEnd:
                boardState = board.getStateAsT() # [1, C, sz, sz]
                actionProbs = self.MCTS.search(board) # [sz*sz]
                pos = self.selectAction(actionProbs, moveCount=len(board.moves))
                if board.placeStone(pos // self.Size, pos % self.Size):
                    states.append(boardState)
                    mctsProbs.append(torch.from_numpy(actionProbs).float())
        result = 1 if board.Winner == 1 else -1 if board.Winner==2 else 0
        # Return Data
        data = []
        for state, mctsProb in zip(states, mctsProbs): # old first
            data.append((state.cpu(), mctsProb.cpu(), result))
            result = -result
        if not (result <= 0): raise ValueError("Wrong result")
        del states, mctsProbs
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return data

    def CreateData(self, numGames:int, numPara:int=4):
        """Create self-play data (support cuda multi-processing)
        :param numGames (int): num of self-play data to be created
        :param numPara (int): num of parallel workers (if > 0)
        """
        if numPara: ### Parallel Self-Play on CUDA
            if not torch.cuda.is_available(): raise RuntimeError("No CUDA available")
            modelDict = self.Net.state_dict()
            args = [(modelDict, torch.device("cuda")) for _ in range(numGames)]
            with multiprocessing.Pool(numPara) as pool:
                allData = pool.starmap(_selfplayWorker, args)
        else: ### Sequential Self-Play (when numPara=0)
            allData = []
            for _ in range(numGames): allData.append(self._SelfPlay())
        
        for data in allData: ### Update Replay Memory
            for state, mctsProb, result in data:
                self.Memory.push(state, mctsProb, result)
        del allData
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def TRAIN(self, lr=1e-3, batch_size=256):
        """Train the model
        :param lr (float): init learning rate
        :param batch_size (int): batch size for training
        """
        num_epochs = 30
        self.Net.train()
        print(f"Learning Rate: {lr},  Batch Size: {batch_size}")
        optimizer = torch.optim.AdamW(self.Net.parameters(), lr=lr, weight_decay=1e-4)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        
        for epoch in range(1, num_epochs+1):
            t_start = time.time()
            
            ### Self-Play # 32 games ~ 2000 moves ~ 15 min for 8 para
            self.Memory.clear() # clear old data
            self.CreateData(48, numPara=4)
            if not (len(self.Memory) > batch_size): continue
            else: print(len(self.Memory))

            ### Sample and Step
            iters = 0 # 迭代次数
            counter = 0 # 无效迭代计数器
            bestLoss = 100
            lossSum = 0.0
            while (counter < 10) and (iters < 200):
                optimizer.zero_grad()
                states_batch, mctsProbs_batch, rewards_batch = self.Memory.sample(batch_size)
                policy, value = self.evalState(states_batch)
                # LOSS: Policy-交叉熵损失 + Value-均方差损失 (+ WeightDecay)
                policyLoss = torch.mean(torch.sum(-mctsProbs_batch * (policy.to(self.dv)), dim=1))
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
                del states_batch, mctsProbs_batch, rewards_batch, policy, value
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            #scheduler.step()
            print(f"Epoch-{epoch},  Iters:{iters},  Time:{(time.time()-t_start)/60:.1f}m")
            print(f"Loss: {lossSum/iters:.4f} ~ {bestLoss:.4f}, {policyLoss.item():.4f} + {valueLoss.item():.4f}")
            if epoch % 10 == 0:
                torch.save(self.Net.state_dict(), rootPath+f"model_{epoch}.pth")
                print("Model Saved")
            print()
        
        print("Training Completed")

def _selfplayWorker(modelDict, device:torch.device):
    """Self-Play Worker (for parallel processing)"""
    #torch.cuda.set_per_process_memory_fraction(0.2)
    player = AlphaDog(device=device) # new player1, same net
    player.Net.load_state_dict(modelDict)
    player.MCTS = MCTS(player.Net, num_simulations=400, device=device)
    return player._SelfPlay()

###
if __name__ == "__main__":
    rootPath = "./"
    multiprocessing.set_start_method('spawn') ###
    print(f"CPUs: {multiprocessing.cpu_count()},  GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available(): print(torch.cuda.get_device_properties(0))
    print()

    player = AlphaDog()
    print(f"{player}  |  {player.dv}")
    player.TRAIN()