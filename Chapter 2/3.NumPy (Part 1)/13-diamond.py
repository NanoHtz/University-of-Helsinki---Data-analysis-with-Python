#!/usr/bin/env python3

import numpy as np

def diamond(n):
	eye1 = np.eye(n, dtype=int)
	eye2 = np.fliplr(eye1)
	top = np.concatenate((eye2, eye1[:, 1:]), axis=1)
	diamond_matrix = np.concatenate((top, top[:-1][::-1]), axis=0)

	return diamond_matrix

def main():
    pass

if __name__ == "__main__":
    main()
