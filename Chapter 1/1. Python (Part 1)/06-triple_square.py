#!/usr/bin/env python3

def triple(x):
	"""Multiplica el argumento por 3"""
	return x * 3

def square(x):
	"""Halla el cuadrado del argumento"""
	return x ** 2

def main():
	for i in range(1, 11, 1):
		t = triple(i)
		s = square(i)
		if s > t:
			break
		print(f"triple({i})=={t} square({i})=={s}")

if __name__ == "__main__":
    main()
