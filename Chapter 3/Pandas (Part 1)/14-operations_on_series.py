#!/usr/bin/env python3

import pandas as pd

def create_series(L1, L2):
	index = ["a", "b", "c"]
	s1 = pd.Series(L1, index)
	s2 = pd.Series(L2, index)
	return (s1, s2)

def modify_series(s1, s2):
	s1["d"] = s2["b"]
	del s2["b"]
	return (s1, s2)

def main():
	L1 = [1, 2, 3]
	L2 = [10, 20, 30]
	s1, s2 = create_series(L1, L2)
	s1, s2 = modify_series(s1, s2)
	result = s1 + s2
	print(result)

if __name__ == "__main__":
    main()
