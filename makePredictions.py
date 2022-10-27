# load libraries
import os
import gc
from numerapi import NumerAPI
from utils.models import RunModel

# instantiate env var
PUBLIC_ID = os.getenv('PUBLIC_ID')
SECRET_KEY = os.getenv('SECRET_KEY')

# instantiate api & check for round
napi = NumerAPI(public_id=PUBLIC_ID, secret_key=SECRET_KEY)
current_round = napi.get_current_round()
print(f"Current round #: {current_round}")

# init class
nmr = RunModel(current_round=current_round)
nmr.get_data()

# run foxhound
nmr.run_foxhound()
gc.collect()

# run deadcell
nmr.run_deadcell()
gc.collect()

# run cobra
nmr.run_cobra()
gc.collect()

# run beautybeast
nmr.run_beautybeast()
gc.collect()

# run skulls
nmr.run_skulls()
gc.collect()

# run desperado
nmr.run_desperadov3()
gc.collect()

# run gaia
nmr.run_gaia()
gc.collect()

# run terra
nmr.run_terra()
gc.collect()

