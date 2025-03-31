#!/usr/bin/env python3

import pandas as pd
import numpy as np

def cities():
	data = np.array([
		[643272, 715.48],
		[279044, 528.03],
		[231853, 689.59],
		[223027, 240.35],
		[201810, 3817.52]
	])
	df = pd.DataFrame(data, columns = ["Population", "Total area"], index = ["Helsinki", "Espoo", "Tampere", "Vantaa", "Oulu"])
	print(df)
	return df

def main():
	cities()
	return

if __name__ == "__main__":
	main()
