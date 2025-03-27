#!/usr/bin/env python3

import pandas as pd

def inverse_series(s):
	inverse = pd.Series(s.index, index=s.values)
	return inverse

def main():
	s = pd.Series({'a': 1, 'b': 2, 'c': 3})
	inverted = inverse_series(s)

	print("Original:")
	print(s)

	print("\nSerie invertida:")
	print(inverted)
	return

if __name__ == "__main__":
    main()
