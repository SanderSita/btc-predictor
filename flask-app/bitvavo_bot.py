import numpy as np
import pandas as pd
import time
import os
import requests
import joblib
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import python_bitvavo_api
from python_bitvavo_api.bitvavo import Bitvavo
from dotenv import load_dotenv

load_dotenv()

# Constants
MODEL_PATH = os.getenv('MODEL_PATH')
SCALER_PATH = os.getenv('SCALER_PATH')
FEATURES = ['High', 'Low', 'Close', 'SMA_21']
SEQUENCE_LENGTH = 30
MAE = 2229  # Model's Mean Absolute Error
MAX_RISK_PERCENTAGE = 0.05  # Maximum risk per trade (5% of portfolio)
TAKE_PROFIT_PERCENTAGE = 0.03  # Take profit at 3% gain
STOP_LOSS_PERCENTAGE = 0.02  # Stop loss at 2% loss

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bitcoin_trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

class BitcoinTradingBot:
    def __init__(self, model_path, scaler_path, api_key="", api_secret=""):
        """Initialize the Bitcoin trading bot with the model and API credentials."""
        logger.info("Initializing Bitcoin Trading Bot")
        # Load the model and scaler
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Initialize Bitvavo API
        self.bitvavo = Bitvavo({
            'APIKEY': api_key,
            'APISECRET': api_secret,
        })
        
        # Trading state
        self.in_position = False
        self.entry_price = 0
        self.position_size = 0
        self.last_prediction = None
        self.eur_usd_rate = self.get_eur_usd_rate()
        logger.info(f"Current EUR/USD rate: {self.eur_usd_rate}")
        
        # Check API connection
        try:
            balance = self.bitvavo.balance({})
            logger.info(f"Connected to Bitvavo API successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Bitvavo API: {e}")
            raise

    def get_eur_usd_rate(self):
        """Get current EUR/USD exchange rate."""
        try:
            # Using a free exchange rate API
            response = requests.get('https://api.exchangerate-api.com/v4/latest/EUR')
            data = response.json()
            rate = data['rates']['USD']
            logger.info(f"Retrieved EUR/USD rate: {rate}")
            return rate
        except Exception as e:
            logger.error(f"Error getting EUR/USD rate: {e}. Using default 1.08")
            # Fallback to a reasonable default if API fails
            return 1.08
        
    def update_exchange_rate(self):
        """Update the EUR/USD exchange rate (call periodically)."""
        try:
            old_rate = self.eur_usd_rate
            self.eur_usd_rate = self.get_eur_usd_rate()
            logger.info(f"Updated EUR/USD rate from {old_rate} to {self.eur_usd_rate}")
        except Exception as e:
            logger.error(f"Failed to update exchange rate: {e}")

    def get_historical_data_from_binance(self):
        """Fetch historical Bitcoin price data in USD from Binance API."""
        try:
            res = requests.get('https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=50')
            data = res.json()
            
            df = pd.DataFrame(data)
            df.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 
                         'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 
                         'Taker Buy Quote Asset Volume', 'Ignore']
            
            df[['High', 'Low', 'Close']] = df[['High', 'Low', 'Close']].astype(float)
            
            # Calculate SMA_21
            df['SMA_21'] = df['Close'].rolling(window=21).mean()
            
            df = df.dropna()
            
            if len(df) >= SEQUENCE_LENGTH:
                df = df.tail(SEQUENCE_LENGTH)
            else:
                logger.warning(f"Not enough data points. Got {len(df)}, need {SEQUENCE_LENGTH}")
            
            return df[FEATURES]
        
        except Exception as e:
            logger.error(f"Error fetching Binance data: {e}")
            logger.info("Falling back to Bitvavo data with EUR/USD conversion")
            return self.get_historical_data_from_bitvavo()

    def get_historical_data_from_bitvavo(self):
        """Fetch historical Bitcoin price data from Bitvavo API and convert to USD."""
        try:
            # Get daily candles for the last 50 days (we need 30 but get extra for SMA calculation)
            candles = self.bitvavo.candles('BTC-EUR', '1d', {'limit': 50})
            
            # Bitvavo returns newest candles first, so we reverse to get chronological order
            candles.reverse()
            
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert to numeric values
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            # Convert EUR to USD using exchange rate
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col] * self.eur_usd_rate
            
            # Rename columns to match model features
            df = df.rename(columns={
                'high': 'High',
                'low': 'Low',
                'close': 'Close'
            })
            
            # Calculate SMA_21
            df['SMA_21'] = df['Close'].rolling(window=21).mean()
            
            # Drop NaN values (from SMA calculation)
            df = df.dropna()
            
            # Get the last 30 days of data (SEQUENCE_LENGTH)
            if len(df) >= SEQUENCE_LENGTH:
                df = df.tail(SEQUENCE_LENGTH)
            else:
                logger.warning(f"Not enough data points. Got {len(df)}, need {SEQUENCE_LENGTH}")
            
            return df[FEATURES]
        
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None

    def get_historical_data(self):
        """Get historical data, preferring Binance's USD data if available."""
        # Try Binance first (USD data)
        binance_data = self.get_historical_data_from_binance()
        if binance_data is not None and len(binance_data) >= SEQUENCE_LENGTH:
            logger.info("Using Binance USD data for prediction")
            return binance_data
        
        # Fallback to Bitvavo with EUR->USD conversion
        logger.info("Falling back to Bitvavo data with EUR/USD conversion")
        return self.get_historical_data_from_bitvavo()

    def get_current_price_usd(self):
        """Get the current Bitcoin price in USD."""
        try:
            # First try from Binance
            res = requests.get('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT')
            data = res.json()
            return float(data['price'])
        except Exception as e:
            logger.error(f"Error getting USD price from Binance: {e}")
            # Fallback to Bitvavo with conversion
            try:
                ticker = self.bitvavo.tickerPrice({'market': 'BTC-EUR'})
                eur_price = float(ticker['price'])
                usd_price = eur_price * self.eur_usd_rate
                logger.info(f"Converted BTC-EUR price {eur_price} to USD: {usd_price}")
                return usd_price
            except Exception as e2:
                logger.error(f"Error getting price from Bitvavo: {e2}")
                return None

    def get_current_price_eur(self):
        """Get the current Bitcoin price in EUR from Bitvavo."""
        try:
            ticker = self.bitvavo.tickerPrice({'market': 'BTC-EUR'})
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Error getting EUR price from Bitvavo: {e}")
            return None

    def predict_next_high(self, df):
        """Predict the next day's high price using the loaded model."""
        try:
            # Scale the data
            scaled_data = self.scaler.transform(df)
            
            # Reshape for model input (add batch dimension)
            X = np.expand_dims(scaled_data, axis=0)
            
            # Make prediction
            predicted_scaled = self.model.predict(X, verbose=0)
            
            # Convert prediction back to original scale
            high_index = FEATURES.index('High')
            
            # Create a dummy array with zeros except for our prediction
            dummy = np.zeros((1, len(FEATURES)))
            dummy[0, high_index] = predicted_scaled[0][0]
            
            # Inverse transform to get the actual predicted value
            predicted_high = self.scaler.inverse_transform(dummy)[0][high_index]
            
            return predicted_high
        
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return None

    def get_account_balance(self):
        """Get current account balance for EUR and BTC."""
        try:
            balances = self.bitvavo.balance({})
            eur_balance = 0
            btc_balance = 0
            
            for balance in balances:
                if balance['symbol'] == 'EUR':
                    eur_balance = float(balance['available'])
                elif balance['symbol'] == 'BTC':
                    btc_balance = float(balance['available'])
            
            return eur_balance, btc_balance
        
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return 0, 0

    def place_buy_order(self, amount_eur):
        """Place a buy order for Bitcoin."""
        try:
            current_price = self.get_current_price_eur()
            if not current_price:
                return False
            
            # Calculate amount of BTC to buy
            btc_amount = amount_eur / current_price
            
            # Format to 8 decimal places (Bitvavo requirement)
            btc_amount = round(btc_amount, 8)
            
            logger.info(f"Placing buy order for {btc_amount} BTC at approximately {current_price} EUR")
            
            # Place market buy order
            order = self.bitvavo.order('BTC-EUR', 'buy', 'market', {
                'amount': str(btc_amount)
            })
            
            logger.info(f"Buy order placed: {order}")
            
            # Update position information
            self.in_position = True
            self.entry_price = current_price
            self.position_size = btc_amount
            
            return True
        
        except Exception as e:
            logger.error(f"Error placing buy order: {e}")
            return False

    def place_sell_order(self, btc_amount):
        """Place a sell order for Bitcoin."""
        try:
            if btc_amount <= 0:
                logger.warning("Cannot sell zero or negative amount")
                return False
                
            logger.info(f"Placing sell order for {btc_amount} BTC")
            
            # Format to 8 decimal places (Bitvavo requirement)
            btc_amount = round(btc_amount, 8)
            
            # Place market sell order
            order = self.bitvavo.order('BTC-EUR', 'sell', 'market', {
                'amount': str(btc_amount)
            })
            
            logger.info(f"Sell order placed: {order}")
            
            # Update position information
            self.in_position = False
            self.entry_price = 0
            self.position_size = 0
            
            return True
        
        except Exception as e:
            logger.error(f"Error placing sell order: {e}")
            return False

    def calculate_position_size(self, risk_amount):
        """Calculate the position size based on risk management."""
        eur_balance, _ = self.get_account_balance()
        
        # Limit the risk to MAX_RISK_PERCENTAGE of the portfolio
        max_risk = eur_balance * MAX_RISK_PERCENTAGE
        
        # Use the smaller of the provided risk amount or max risk
        actual_risk = min(risk_amount, max_risk)
        
        return actual_risk

    def execute_strategy(self):
        """Execute the trading strategy based on model prediction."""
        # Get current prices
        current_price_usd = self.get_current_price_usd()
        current_price_eur = self.get_current_price_eur()
        
        if not current_price_usd or not current_price_eur:
            logger.error("Could not get current prices")
            return
        
        # Get historical data
        df = self.get_historical_data()
        if df is None or len(df) < SEQUENCE_LENGTH:
            logger.error(f"Not enough data points for prediction. Need {SEQUENCE_LENGTH}")
            return
        
        # Make prediction (in USD)
        predicted_high_usd = self.predict_next_high(df)
        if predicted_high_usd is None:
            logger.error("Failed to make prediction")
            return
        
        # Calculate signal based on prediction vs current price (in USD)
        delta_usd = predicted_high_usd - current_price_usd
        
        logger.info(f"Current price (USD): {current_price_usd}")
        logger.info(f"Current price (EUR): {current_price_eur}")
        logger.info(f"Predicted high for next day (USD): {predicted_high_usd}")
        logger.info(f"Delta (USD): {delta_usd}")
        
        # Store the last prediction
        self.last_prediction = predicted_high_usd
        
        # Check if we're already in a position
        if self.in_position:
            # Check if we should take profit or cut losses (using EUR price for trading)
            profit_percentage = (current_price_eur - self.entry_price) / self.entry_price
            
            if profit_percentage >= TAKE_PROFIT_PERCENTAGE:
                logger.info(f"Taking profit at {profit_percentage:.2%}")
                _, btc_balance = self.get_account_balance()
                self.place_sell_order(btc_balance)
                
            elif profit_percentage <= -STOP_LOSS_PERCENTAGE:
                logger.info(f"Stopping loss at {profit_percentage:.2%}")
                _, btc_balance = self.get_account_balance()
                self.place_sell_order(btc_balance)
                
            # Also check our prediction - if it's now bearish, consider exit
            elif delta_usd < -MAE:
                logger.info("Prediction turned bearish, exiting position")
                _, btc_balance = self.get_account_balance()
                self.place_sell_order(btc_balance)
        
        else:  # Not in position
            # Buy signal: predicted high is significantly above current price
            if delta_usd > MAE:
                logger.info("Bullish signal detected, entering position")
                
                # Calculate position size (how much EUR to use)
                eur_balance, _ = self.get_account_balance()
                position_eur = self.calculate_position_size(eur_balance * 0.5)  # Use 50% of available balance, subject to risk limits
                
                if position_eur >= 10:  # Minimum order size for Bitvavo is typically around 10 EUR
                    self.place_buy_order(position_eur)
                else:
                    logger.info(f"Position size too small: {position_eur} EUR")
    
    def run(self, interval_minutes=60):
        """Run the trading bot at specified intervals."""
        logger.info(f"Starting Bitcoin Trading Bot")
        logger.info(f"Checking for trades every {interval_minutes} minutes")
        
        while True:
            try:
                # Update exchange rate once per cycle
                self.update_exchange_rate()
                
                # Print account balance
                eur_balance, btc_balance = self.get_account_balance()
                current_price_eur = self.get_current_price_eur()
                current_price_usd = self.get_current_price_usd()
                
                if current_price_eur:
                    total_balance_eur = eur_balance + (btc_balance * current_price_eur)
                    logger.info(f"Account Balance: {eur_balance:.2f} EUR + {btc_balance:.8f} BTC = {total_balance_eur:.2f} EUR")
                    
                    if current_price_usd:
                        logger.info(f"BTC/EUR: {current_price_eur} | BTC/USD: {current_price_usd} | EUR/USD rate: {self.eur_usd_rate}")
                
                # Execute trading strategy
                self.execute_strategy()
                
                # Log the next check time
                next_check = datetime.now() + timedelta(minutes=interval_minutes)
                logger.info(f"Next check at {next_check.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Sleep until next check
                time.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                # Sleep for a minute
                time.sleep(60)

if __name__ == "__main__":
    API_KEY = os.getenv('API_KEY')
    API_SECRET = os.getenv('API_SECRET')
    
    bot = BitcoinTradingBot(
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        api_key=API_KEY,
        api_secret=API_SECRET
    )
    
    # checking for trades every hour
    bot.run(interval_minutes=60)