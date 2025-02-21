#!/usr/bin/env python3

import math

def solve_quadratic(a, b, c):
	q = float(math.sqrt(b**2 - (4*a*c)))
	x = float((-b + q) / (2*a))
	y = float((-b - q) / (2*a))
	return (x,y)


def main():
    solve_quadratic(1, 2, 1)

if __name__ == "__main__":
    main()
