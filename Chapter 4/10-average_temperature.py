#!/usr/bin/env python3

import pandas as pd

def average_temperature():
	df = pd.read_csv("src/kumpula-weather-2017.csv", sep=",")
	july = df[df['m'] == 7]
	media = july['Air temperature (degC)'].mean()
	return media

def main():
	media = average_temperature()
	print(f"Average temperature in July: {media:.1f}")

if __name__ == "__main__":
	main()
