#!/usr/bin/env python3

import numpy as np
from functools import reduce

def matrix_power(a, n):
	m = a.shape[0]
	if n == 0:
		return np.eye(m)
	elif n > 0:
		return reduce(lambda x, y: x @ y, (a for _ in range(n)))
	else:
		inv_a = np.linalg.inv(a)
		return reduce(lambda x, y: x @ y, (inv_a for _ in range(-n)))

def main():
	a = np.array([[2, 0],
				[0, 2]])

	print(matrix_power(a, 3))
	print(matrix_power(a, 0))
	print(matrix_power(a, -1))


if __name__ == "__main__":
    main()
