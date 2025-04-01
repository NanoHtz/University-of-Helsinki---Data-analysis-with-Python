#!/usr/bin/env python3

import pandas as pd
import numpy as np

def last_week():
	df = pd.read_csv("src/UK-top40-1964-1-2.tsv", sep="\t")
	df["WoC"] -= 1
	df = df[~df["LW"].isin(["New", "Re"])]
	df["LW"] = df["LW"].astype(int)
	is_peak = df["Peak Pos"] == df["Pos"]
	is_valid = df["Pos"] < df["LW"]
	df.loc[is_peak & is_valid, "Peak Pos"] = np.nan
	df = df.sort_values(by="LW")
	df.index = df["LW"]
	df = df.reindex(range(1, 41))
	df["Pos"] = df.index
	df["LW"] = np.nan
	return df

def main():
	df = last_week()
	print("Shape: {}, {}".format(*df.shape))
	print("dtypes:", df.dtypes)
	print(df)

if __name__ == "__main__":
	main()
