#!/usr/bin/env python3

import pandas as pd

def below_zero():
	df = pd.read_csv("src/kumpula-weather-2017.csv", sep=",")
	zero = len(df[df['Air temperature (degC)'] < 0])
	return zero

def main():
	zero = below_zero()
	print(f"Number of days below zero: {zero}")
	return

if __name__ == "__main__":
	main()
