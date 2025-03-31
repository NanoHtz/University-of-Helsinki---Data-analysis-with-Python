#!/usr/bin/env python3

import pandas as pd

def municipalities_of_finland():
	df = pd.read_csv("src/municipal.tsv", sep="\t", index_col="Region 2018")
	df = df.loc["Akaa":"Äänekoski"]
	return df

def main():
	df = municipalities_of_finland()
	print (df)
	return

if __name__ == "__main__":
    main()
