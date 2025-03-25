#!/usr/bin/env python3

import numpy as np

def first_half_second_half(a):
	m = a.shape[1] // 2
	sum_first = np.sum(a[:, :m], axis=1)
	sum_second = np.sum(a[:, m:], axis=1)
	return a[sum_first > sum_second]

def main():
	a = np.array([[1, 3, 4, 2],
				  [2, 2, 1, 2]])
	print(first_half_second_half(a))

if __name__ == "__main__":
    main()
