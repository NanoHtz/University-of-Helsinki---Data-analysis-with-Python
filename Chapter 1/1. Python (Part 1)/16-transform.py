#!/usr/bin/env python3

def transform(s1, s2):
	n1 = map(int, s1.split())
	n2 = map(int, s2.split())
	result = [a * b for a, b in zip(n1, n2)]

	return result

def main():
    pass

if __name__ == "__main__":
    main()
