# Lubibot gives a trading framework
from lumibot.brokers import Alpaca # Broker
from lumibot.backtesting import YahooDataBacktesting # Dataframe
from lumibot.strategies.strategy import Strategy # Actual TradingBot
from lumibot.traders import Trader # Deploment Capability
from datetime import datetime 
from alpaca_trade_api import REST # get data dynamically
from timedelta import Timedelta # calculate easier difference between weeks and days
from finbert_utils import estimate_sentiment # ml model

#TODO create a file to import the APIs, so that it can be included in the '.gitignore'
# Make sure when getting the API key, to save it in a save place, since is not going to appear again
API_KEY = "YOUR API KEY" 
API_SECRET = "YOUR API SECRET" 
BASE_URL = "https://paper-api.alpaca.markets"

ALPACA_CREDS = {
    "API_KEY":API_KEY, 
    "API_SECRET": API_SECRET, 
    "PAPER": True
}

# Framework for the strategy. This is the backbone of the trading bot
class MLTrader(Strategy): 
    # This will run once (lifecycle method)
    def initialize(self,
                   symbol:str="SPY", # is using the SPY Index
                   cash_at_risk:float=.5): # how much cash to risk by default
        self.symbol = symbol
        # Here is dictated how frequently is going to trade
        self.sleeptime = "24H" 
        # Capture our last trade to undo some bias or ¿selfs?
        self.last_trade = None 
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

    # To make the position size dynamic
    def position_sizing(self): 
        cash = self.get_cash() 
        last_price = self.get_last_price(self.symbol)
        # Here is the sizing calculated
        # This guides how much of our cash balance we use per trade
        # cash_at_risk of 0.5 means that for each trade we're using 50% of our remainig cash balance
        quantity = round(cash * self.cash_at_risk / last_price,0)
        return cash, last_price, quantity

    # get the dynamic date, based on when the trade is done
    def get_dates(self): 
        today = self.get_datetime() # with respect to the backtest
        three_days_prior = today - Timedelta(days=3)
        # resturns '%Y-%m-%d' as a string
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self): 
        # get sentiments for the evaluated day
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, 
                                 start=three_days_prior, 
                                 end=today) 
        # loops through each value to get dates and news
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment 

    # Is going to run everytime it gets a tick (lifecycle method)
    def on_trading_iteration(self):
        # get position size
        cash, last_price, quantity = self.position_sizing() 
        # get sentiment and probability
        probability, sentiment = self.get_sentiment()

        # to avoid buying when we don't have cash
        if cash > last_price: 
            # This is for buying
            if sentiment == "positive" and probability > .999: 
                # this is for if there is an existing sell order
                if self.last_trade == "sell": 
                    self.sell_all() 
                order = self.create_order(
                    self.symbol, 
                    quantity, 
                    "buy", 
                    # here are the limits of the buy specified
                    type="bracket", 
                    take_profit_price=last_price*1.20, # 20%
                    stop_loss_price=last_price*.95 # 5%
                )
                self.submit_order(order) 
                self.last_trade = "buy"
            # This is for selling
            elif sentiment == "negative" and probability > .999: 
                if self.last_trade == "buy": 
                    self.sell_all() 
                order = self.create_order(
                    self.symbol, 
                    quantity, 
                    "sell", 
                    # here are the limits of the sell specified
                    type="bracket", 
                    take_profit_price=last_price*.8, # 20%
                    stop_loss_price=last_price*1.05 # 5%
                )
                self.submit_order(order) 
                self.last_trade = "sell"

# Times
start_date = datetime(2020,1,1)
end_date = datetime(2023,12,31) 

broker = Alpaca(ALPACA_CREDS) 
# 'mlstrat' is arbitrary
strategy = MLTrader(name='mlstrat', broker=broker, 
                    # Here is the Index specified and the Risk to take
                    parameters={"symbol":"SPY", 
                                "cash_at_risk":.5})
# Evaluate how well this may run
strategy.backtest(
    YahooDataBacktesting, 
    start_date, 
    end_date, 
    # This should be modified together with the previous one
    parameters={"symbol":"SPY", "cash_at_risk":.5}
)

# # This is to deploy live
# # connect trader
# trader = Trader()
# # add strategy to trader
# trader.add_strategy(strategy)
# trader.run_all()
