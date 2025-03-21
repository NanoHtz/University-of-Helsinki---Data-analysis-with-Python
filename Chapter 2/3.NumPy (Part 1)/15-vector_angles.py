#!/usr/bin/env python3

import numpy as np
import scipy.linalg

def vector_lengths(a):
	b = a**2
	c = np.sum(b, axis = 1)
	d = np.sqrt(c)
	return d

def vector_angles(X, Y):
	dot = np.sum(X * Y, axis=1)
	x = vector_lengths(X)
	y = vector_lengths(Y)
	cos = dot/(x*y)
	cos = np.clip(cos, -1.0, 1.0)
	rad = np.arccos(cos)
	deg = np.degrees(rad)
	return deg

def main():
	X = np.array([[1, 0], [0, 1]])
	Y = np.array([[0, 1], [1, 0]])
	angles = vector_angles(X, Y)
	print(angles)

if __name__ == "__main__":
	main()
