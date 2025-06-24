from dotenv import load_dotenv
import os
from balldontlie import BalldontlieAPI

class Client:
    """Base client for accessing the Balldontlie API."""
    
    def __init__(self, api_key=None):
        # Load from .env file if api_key not provided
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("BALLDONTLIE_API_KEY")
            
        if api_key is None:
            raise ValueError("API key is required. Either pass it directly or set BALLDONTLIE_API_KEY in your environment.")
        
        self.api = BalldontlieAPI(api_key=api_key)


class SportsClient(Client):
    """Client for accessing sports data through the Balldontlie API."""
    
    def __init__(self, api_key=None):
        super().__init__(api_key)
        
    @property
    def nba(self):
        """Access NBA data through the Balldontlie API."""
        return self.api.nba
    
    @property
    def mlb(self):
        """Access MLB data through the Balldontlie API."""
        return self.api.mlb
    
    @property
    def nfl(self):
        """Access NFL data through the Balldontlie API."""
        return self.api.nfl
        
    @property
    def epl(self):
        """Access EPL data through the Balldontlie API."""
        return self.api.epl