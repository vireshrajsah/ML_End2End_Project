import os
import pickle
from src.utils import *

print(os.getcwd())

model = load_object(os.path.join('artifacts', 'model.pkl'))

print(type(model))