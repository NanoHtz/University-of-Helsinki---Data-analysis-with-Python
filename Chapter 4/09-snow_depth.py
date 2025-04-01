#!/usr/bin/env python3

import pandas as pd

def snow_depth():
	df = pd.read_csv("src/kumpula-weather-2017.csv", sep=",")
	df['Date'] = pd.to_datetime(df[['Year', 'm', 'd']].rename(columns={'Year': 'year', 'm': 'month', 'd': 'day'}))
	df_valid_snow = df[df['Snow depth (cm)'] != -1]
	max_snow = df_valid_snow['Snow depth (cm)'].max()
	return max_snow

def main():
	snow = snow_depth()
	print(f"Max snow depth: {snow}")
	return

if __name__ == "__main__":
	main()
