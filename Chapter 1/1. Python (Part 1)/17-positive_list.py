#!/usr/bin/env python3

def positive(x):
	return x > 0

def positive_list(L):
	result = list(filter(positive, L))
	return result

def main():
    pass

if __name__ == "__main__":
    main()
