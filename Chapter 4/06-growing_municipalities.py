#!/usr/bin/env python3

import pandas as pd

def growing_municipalities(df):
	growing = df["Population change from the previous year, %"] > 0
	proportion = growing.sum() / len(df)
	return proportion

def main():
	df = pd.read_csv("src/municipal.tsv", sep="\t", index_col="Region 2018")
	df = df.loc["Akaa":"Äänekoski"]
	print(f"Proportion of growing municipalities: {growing_municipalities(df)*100:.1f}%")
	return

if __name__ == "__main__":
    main()
