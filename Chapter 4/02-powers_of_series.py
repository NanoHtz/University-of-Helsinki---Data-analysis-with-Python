#!/usr/bin/env python3

import pandas as pd

def powers_of_series(s, k):
	data = {i: s**i for i in range(1, k+1)}
	df = pd.DataFrame(data, index=s.index)
	return df

def main():
	s = pd.Series([1, 2, 3, 4], index=list("abcd"))
	print(powers_of_series(s, 5))

if __name__ == "__main__":
    main()
