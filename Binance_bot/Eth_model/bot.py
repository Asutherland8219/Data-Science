import websocket

## <symbol> is where you put the asset you want to trade, in this instance its ETHUSD
## <interval> is the time interval between prices 

SOCKET = "wss://stream.binance.com:9443/ws/ethusdt@kline_1m"

## btcusdt is bitcoin, ltcusdt is litecoin, 

def on_open(ws):
    print('opened connection')

def on_close(ws):
    print('closed_connection')

def on_message(ws, message):
    print('received message')

ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
ws.run_forever()



