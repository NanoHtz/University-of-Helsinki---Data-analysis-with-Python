#!/usr/bin/env python3

import pandas as pd

def read_series():
	s = pd.Series(dtype=object)

	while True:
		line = input()
		if not line:
			break
		parts = line.split()
		if len(parts) != 2:
			raise Exception("Entrada incorrecta")
		index, value = parts
		new = pd.Series({index : value})
		s = pd.concat([s, new])
	return s

def main():
	series = read_series()
	print(series)
	pass

if __name__ == "__main__":
    main()
