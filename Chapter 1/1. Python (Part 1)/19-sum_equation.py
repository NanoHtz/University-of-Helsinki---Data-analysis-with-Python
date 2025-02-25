#!/usr/bin/env python3

def sum_equation(L):
	if L == []:
		return "0 = 0"
	else:
		new_str = " + ".join(map(str, L)) + " = " + str(sum(L))
	return new_str

def main():
	new_str = sum_equation([1, 2, 3, 4, 5])
	print(new_str)
	


if __name__ == "__main__":
    main()
