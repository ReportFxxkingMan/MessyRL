import pyupbit
import gym
import pandas as pd
from typing import Tuple

class UpbitEnv(gym.Env):
    def __init__(self, init_balance : int, n_window : int, interval : str = "minute240", count : int = 1000) -> None:
        '''
        Initialize self.
        
        Load dataset
        
        Parameters
        ---------------
        init_balance : initial balance (Won)
        
        n_window : the # of window (previous time units) for each observation (state)
        
        interval : time unit (Default: minute240)
        
        count : the # of number of observations in total data (Default: 1000)
        
        '''
        super(UpbitEnv, self).__init__()
        
        data_eth = pyupbit.get_ohlcv("KRW-ETH", interval = interval, count = count)

        data_eth['candle'] = data_eth['close'] - data_eth['open']
        data_eth = data_eth[['close', 'volume', 'candle']] 
        
        self._scaler_dict = {'volume' : [data_eth['volume'].mean(), data_eth['volume'].std()],
                            'candle' : [data_eth['candle'].mean(), data_eth['candle'].std()]}
       
        # 관측치 (state) 로 사용되는 변수들 정규화 
        for col in ['volume', 'candle']:
            data_eth[col] = (data_eth[col] - self._scaler_dict[col][0] )/ self._scaler_dict[col][1] 
        
        self.data = data_eth
    
        self.n_window = n_window
        
        self.init_balance = init_balance 
         
        
        self.observation_space = gym.spaces.Box(low = - 100.0, high = 100.0, shape = (n_window, 2))
        self.action_space = gym.spaces.Discrete(2)

        self.n = self.data.shape[0]
        
    def reset(self) -> list : 
        '''
        reset Environment.
        
        Load dataset
        
        Returns
        ----------------
        obs : list
            An observation of the first of this evironment.
        
        '''
        self.current_time = self.n_window
        self.balance = self.init_balance
        
        self.coin_balance = 0
        self.done = False
        
        self.pre_balance = self.init_balance
        
        obs =self.data.iloc[(self.current_time- self.n_window):self.current_time,:][ ['candle', 'volume']]
        
        obs = obs.values.tolist()
        
        return obs
    
    def step(self, action : int) -> Tuple[list, float, bool, dict]:
        
        '''
        Get state, reward from an action. 
        
        Parameters
        ---------------
        action : number of action (0 : sell ETH, 1: buy ETH)
        
        Returns
        ---------------
        obs : list
            State changed by an action from current state.
        
        reward : float
            Rewards for an action in the current state.
        
        done : bool
            Boolean about the end of this environment.
        
        remain_info : dictionary
            Additional information about the current state.
            This is empty in this environment but is intended to be formatted for using StableBaselines3.
        
        '''
        
        price = self.data['close'][self.current_time]
        
        obs =self.data.iloc[(self.current_time- self.n_window):self.current_time,:][ ['candle', 'volume']]
        obs = obs.values.tolist()
        
        if action == 0 :
            self.sell(price)
        elif action == 1 :
            self.buy(price)
            
        # 현재 가치 평가 (원화 기준)
        current_balance = self.balance + self.coin_balance*price
    
        # 리워드 (이전에 비한 수익, 원화 기준 )
        reward = (current_balance - self.pre_balance)/self.pre_balance
        
        self.pre_balance = current_balance
        
        # 시간 흐름   
        self.current_time += 1       
        
        # 종료 조건
        if self.current_time == self.n:
            self.done = True
            
        remain_info = {}
        
        return obs, reward, self.done, remain_info
        
    # 코인 판매
    def sell(self, price : int):
        '''
        Sell all ETHs.

        Parameters
        ----------
        price : int
            Current price of ETH.

        Returns
        -------
        None.

        '''
        
        self.balance += price*self.coin_balance
        self.coin_balance = 0
        
    # 코인 구매
    def buy(self, price : int):
        
        '''
        Buy all ETHs.

        Parameters
        ----------
        price : int
            Current price of ETH.

        Returns
        -------
        None.

        '''
        
        self.coin_balance += self.balance/price
        self.balance = 0
        
    def render(self, mode='human'):
        '''
        This is empty in this environment but is intended to be formatted for using StableBaselines3.
        '''
        
        return None
    
    def close (self):
        '''
        This is empty in this environment but is intended to be formatted for using StableBaselines3.
        '''
        
        return None    