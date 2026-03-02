from os.path import dirname, abspath
directorio = dirname(dirname(abspath(__file__)))
import sys
sys.path.append(directorio)
from src import *