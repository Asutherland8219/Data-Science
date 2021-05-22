import websocket, json, pprint, talib, numpy
import config
from binance.client import Client
from binance.enums import *
## <symbol> is where you put the asset you want to trade, in this instance its ETHUSD
## <interval> is the time interval between prices 

SOCKET = "wss://stream.binance.com:9443/ws/ethusdt@kline_1m"
RSI_Period = 14
# RSI_Period = 30  , for BTC in a previous test, RSI 30 was the strongest correlated predictable variable

RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
TRADE_SYMBOL = 'ETHUSD'
TRADE_QUANTITY = 0.05

closes = []
in_position = False

client = Client(config.API_KEY, config.API_SECRET, tld='cad')
## btcusdt is bitcoin, ltcusdt is litecoin, 

def order(side, quantity, symbol, order_type=ORDER_TYPE_MARKET):
    try:
        print("sending order")
        order = client.create_order(symbol=symbol,
        side=side,
        type=order_type,
        quantity=quantity)
        print(order)
    except Exception as e:
        return False
    
    return True

def on_open(ws):
    print('opened connection')

def on_close(ws):
    print('closed_connection')

def on_message(ws, message):
    print('received message')
    json_message = json.loads(message)
    pprint.pprint(json_message)

    candle = json_message['k']
    is_candle_closed = candle['x']
    close = candle['c']


    if is_candle_closed:
        print('candle closed at {}'.format(close))
        closes.append(float(close))
        print('closes')
        print(closes)

        if len(closes) > RSI_Period:
            np_closes = numpy.arry(closes)
            rsi = talib.RSI(np_closes, RSI_Period)
            print("all rsis calculated so far")
            print(rsi)
            last_rsi = rsi[-1]
            print("the current rsi is {}".format(last_rsi))

            if last_rsi > RSI_OVERBOUGHT:
                print("Sell! Sell! Sell!")
                order_succeeded = order(SIDE_SELL, TRADE_QUANTITY, TRADE_SYMBOL)
                if order_succeeded:
                    in_positions = False
            else: 
                print("It is overbought, but we dont own any.")

            if last_rsi < RSI_OVERSOLD:
                if in_position:
                    print("It is oversold, but you already own it, nothing to do")
                else:
                    print("Buy! Buy! Buy!")
                    order_succeeded = order(SIDE_BUY, TRADE_QUANTITY, TRADE_SYMBOL)
                if order_succeeded:
                    in_positions = True
                    







ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
ws.run_forever()



