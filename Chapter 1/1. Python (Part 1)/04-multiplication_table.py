#!/usr/bin/env python3


def main():
	for i in range(1,11,1):
		for x in range(1,11,1):
			print(f"{x*i:4}", end="")
		print()

if __name__ == "__main__":
    main()
