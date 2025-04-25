from datetime import datetime
import time
from realtime_crypto import RealTimeCrypto

tracker = RealTimeCrypto()

# Get bitcoin price
bitcoin = tracker.get_coin("bitcoin")
price = bitcoin.get_price()
print(price)

# Get price history
history = bitcoin.get_history("1Y")

high24h = bitcoin.get_statistics().high24h
eth_1h = bitcoin.get_statistics().get_price_change("1h")
eth_24h = bitcoin.get_statistics().get_price_change("24h")
eth_1y = bitcoin.get_statistics().get_price_change("1y")

# Get fear and greed index

a = tracker.get_current_fear_greed_index()
print(a)

# get unix time from january 1, 2019 to january 1, 2021
unix_time = int(datetime(2019, 1, 1).timestamp())
unix_time_now = int(datetime(2021, 1, 1).timestamp())
w = tracker.get_fear_greed_index_history(unix_time, unix_time_now)

# Get best performing cryptos
best = tracker.get_best_performing_cryptos("24h")
print("rtgrt")


import asyncio

# Get realtime crypto prices
async def main():
    async def callback(data):
        new_price = data.get_new_price()
        # name = ws_detail.get_crypto()
        # print(f"{name}: {new_price}")
        print(f"New price: {new_price}")

    # Track bitcoin price
    bitcoin = tracker.get_coin("bitcoin")
    asyncio.create_task(bitcoin.get_realtime_price(callback))

    # Track multiple coins
    # coins = ["ethereum", "dogecoin", "pepe"]
    # asyncio.create_task(tracker.realtime_prices(coins, callback))

    # Ensure script doesn't close
    await asyncio.sleep(1000)

asyncio.run(main())