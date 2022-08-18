import pyupbit
import gym
import pandas as pd


class Upbit_Env(gym.Env):
    def __init__(self, init_account : int, n_window : int, interval : str = "minute240", count : int = 1000):
        super(Upbit_Env, self).__init__()
        
        data_ETH = pyupbit.get_ohlcv("KRW-ETH", interval = interval, count = count)

        data_ETH['candle'] = data_ETH['close'] - data_ETH['open']
        data_ETH = data_ETH[['close', 'volume', 'candle']] 
        
        self._scaler_dict = {'volume' : [data_ETH['volume'].mean(), data_ETH['volume'].std()],
                            'candle' : [data_ETH['candle'].mean(), data_ETH['candle'].std()]}
       
        # 관측치 (state) 로 사용되는 변수들 정규화 
        for col in ['volume', 'candle']:
            data_ETH[col] = (data_ETH[col] - self._scaler_dict[col][0] )/ self._scaler_dict[col][1] 
        
        self.data = data_ETH
    
        self.n_window = n_window
        
        self.init_account = init_account 
         
        
        self.observation_space = gym.spaces.Box(low = - 100.0, high = 100.0, shape = (n_window+1, 5))
        self.action_space = gym.spaces.Discrete(2)

        self.n = self.data.shape[0]
        
    def reset(self):
        self.current_time = self.n_window
        self.account = self.init_account
        
        self.coin_account = 0
        self.done = False
        
        self.pre_account = self.init_account
        
        obs =self.data.iloc[(self.current_time- self.n_window):self.current_time,:][ ['candle', 'volume']]
        
        obs = obs.values.tolist()
        
        return obs
    
    def step(self, action : int):
        
        price = self.data['close'][self.current_time]
        
        obs =self.data.iloc[(self.current_time- self.n_window):self.current_time,:][ ['candle', 'volume']]
        obs = obs.values.tolist()
        
        if action == 0 :
            self.sell(price)
        elif action == 1 :
            self.buy(price)
            
        # 현재 가치 평가 (원화 기준)
        current_account = self.account + self.coin_account*price
    
        # 리워드 (이전에 비한 수익, 원화 기준 )
        reward = (current_account - self.pre_account)/self.pre_account
        
        self.pre_account = current_account
        
        # 시간 흐름   
        self.current_time += 1       
        
        # 종료 조건
        if self.current_time == self.n:
            self.done = True
            
        
        return obs, reward, self.done, {}
        
    # 코인 판매
    def sell(self, price : int):
        self.account += price*self.coin_account
        self.coin_account = 0
        
    # 코인 구매
    def buy(self, price : int):
        self.coin_account += self.account/price
        self.account = 0
        
    def render(self, mode='human'):
        return None
    
    def close (self):
        return None    
    