#!/usr/bin/env python3

import sys
import math

def summary(filename):
	count = 0
	sum = 0.0
	dev = 0.0
	with open(filename, "r") as file:
		for line in file:
			try:
				num = float(line.strip())
				count += 1
				sum += num
				dev += num**2
			except ValueError:
				pass
	if count == 0:
		return(0.0, 0.0, 0.0)
	prom = sum /count
	stddev = math.sqrt((dev - count * prom ** 2) / (count - 1))
	return (sum,prom,stddev)

def main():
	for filename in sys.argv[1:]:
		suma, prom, stddev = summary(filename)
		print(f"File: {filename} Sum: {suma:.6f} Average: {prom:.6f} Stddev: {stddev:.6f}")

if __name__ == "__main__":
    main()
