# load libraries
import pandas as pd
from numerapi import NumerAPI

napi = NumerAPI(public_id, secret_key)
current_round = napi.get_current_round()
print(f"Current round #: {current_round}")