#!/usr/bin/env python3

import numpy as np
import scipy.linalg

def vector_lengths(a):
	b = a**2
	c = np.sum(b, axis = 1)
	d = np.sqrt(c)
	return d

def main():
	data = np.array([[3, 4], [5, 12]])
	result = vector_lengths(data)
	print(result)

if __name__ == "__main__":
    main()
