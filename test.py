import cv2
import os
import pickle
import sys
import numpy as np
import glob
import math


r = np.array([[1, 2], [4, 5]])
r2 = np.array([[6, 7], [8, 9]])
a = np.abs((r-r2))

print(a)
#print(np.diff)
print(np.abs(np.diff(a, axis=0)))

