#!/usr/bin/env python3

import pandas as pd

def split_date_continues():
	df = pd.read_csv("src/Helsingin_pyorailijamaarat.csv", sep= ";")

	df = df.dropna(how="all")
	df = df.dropna(axis = 1, how="all")

	split_columns = df['Päivämäärä'].str.split(expand=True)
	split_columns.columns = ['Weekday', 'Day', 'Month', 'Year', 'Hour']
	split_columns['Day'] = split_columns['Day'].astype(int)
	split_columns['Year'] = split_columns['Year'].astype(int)
	split_columns['Hour'] = split_columns['Hour'].str.split(":").str[0].astype(int)
	months = {
		'tammi': 1, 'helmi': 2, 'maalis': 3, 'huhti': 4,
		'touko': 5, 'kesä': 6, 'heinä': 7, 'elo': 8,
		'syys': 9, 'loka': 10, 'marras': 11, 'joulu': 12
		}
	split_columns['Month'] = split_columns['Month'].map(months).astype(int)

	df = df.drop(columns=['Päivämäärä'])

	df = pd.concat([split_columns, df], axis =1)
	return df

def main():
	df = split_date_continues()
	print("Shape:", df.shape)
	print("Column names:\n", df.columns)
	print(df.head())


if __name__ == "__main__":
	main()
