#!/usr/bin/env python3

class Prepend(object):
	"""Asi se crea una clase, que debe estar basada en una clase previa(object)"""
	def __init__ (self, start):
		"""Inicializamos la clase, el primer parametro debe ser siempre self, lo demas seran argumentos, que querras que el usuario pase al objeto"""
		self.start = start

	def write(self, s):
		"""Escribe la concatenacion de la cadena inicial con lo introducido en el método"""
		print(self.start + s)

def main():
	pass

if __name__ == "__main__":
	main()
