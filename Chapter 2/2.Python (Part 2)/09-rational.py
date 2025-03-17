#!/usr/bin/env python3

class Rational(object):
	def __init__(self, num, den):
		self.num = num
		self.den = den

	def __add__(self, other):
		new_num = self.num * other.den + self.den * other.num
		new_den = self.den * other.den
		return Rational(new_num, new_den)

	def __sub__(self, other):
		new_num = self.num * other.den - self.den * other.num
		new_den = self.den * other.den
		return Rational(new_num, new_den)

	def __mul__(self, other):
		new_num = self.num * other.num
		new_den = self.den * other.den
		return Rational(new_num, new_den)

	def __truediv__(self, other):
		new_num = self.num * other.den
		new_den = self.den * other.num
		return Rational(new_num, new_den)

	def __eq__(self, other):
		return self.num * other.den == self.den * other.num

	def __lt__(self, other):
		return self.num * other.den < self.den * other.num

	def __gt__(self, other):
		return self.num * other.den > self.den * other.num

	def __str__(self):
		return f"{self.num}/{self.den}"



def main():
    r1=Rational(1,4)
    r2=Rational(2,3)
    print(r1)
    print(r2)
    print(r1*r2)
    print(r1/r2)
    print(r1+r2)
    print(r1-r2)
    print(Rational(1,2) == Rational(2,4))
    print(Rational(1,2) > Rational(2,4))
    print(Rational(1,2) < Rational(2,4))

if __name__ == "__main__":
    main()
