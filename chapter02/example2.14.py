import math
from os import getcwd
import datetime

sqrt_value = math.sqrt(16)
current_dir = getcwd()
now = datetime.datetime.now()
print(sqrt_value, current_dir, now, sep="\n")