#!/usr/bin/env python3

def main():
	L = [(x, y) for x in range(1,7,1) for y in range(1,7,1) if x + y == 5]

	for pair in L:
		print(pair)

if __name__ == "__main__":
	main()
