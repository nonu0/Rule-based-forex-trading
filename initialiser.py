import MetaTrader5 as mt5


class MetaTrader5Client:
    # show Metatrader info
    
    print('MetaTrader5 package author: ',mt5.__author__)
    print('MetaTrader5 package version: ',mt5.__version__)
    
    def __init__(self,account,password,server):
        self.account = account
        self.password = password
        self.server = server
        
    def login(self):
        if not mt5.initialize():
            print("initialize() failed, error code =",mt5.last_error())
            quit()
            
        authorized = mt5.login(self.account,self.password,self.server)
        if authorized:
            print("login successful")
            
        else:
            print(f"Failed to connect to account #{self.account}, error code: {mt5.last_error()}")
            
    def shutdown(self):
        mt5.shutdown()