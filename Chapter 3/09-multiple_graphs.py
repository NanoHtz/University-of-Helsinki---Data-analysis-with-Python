#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def main():
	x = [2, 4, 6, 7]
	y = [4, 3, 5, 1]
	X = [1, 2, 3, 4]
	Y = [4, 2, 3, 1]
	plt.plot(x, y)
	plt.plot(X, Y)
	plt.title("Comparison")
	plt.xlabel("x-X")
	plt.ylabel("y-Y")
	plt.show()
	pass

if __name__ == "__main__":
    main()
