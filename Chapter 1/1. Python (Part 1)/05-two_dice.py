#!/usr/bin/env python3

def main():
	for i in range(1,7,1):
		for x in range(1,7,1):
			if x + i == 5:
				print(f"({x},{i})", end="")
				print()

if __name__ == "__main__":
    main()
