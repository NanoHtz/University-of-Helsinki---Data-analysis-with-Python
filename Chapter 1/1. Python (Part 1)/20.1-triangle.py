# Enter you module contents here
"""
Módulo triangle.py

Este módulo proporciona funciones para calcular:
- La hipotenusa de un triángulo rectángulo.
- El área de un triángulo rectángulo.

"""
import math

__author__ = "Fernando Gálvez Gorbe"
__version__ = "1.0.0"
__description__ = "Módulo para calcular la hipotenusa y el área de un triángulo rectángulo."

def hypotenuse(a,b):
	"""Cálcula la hipotenusa dados dos lados"""
	return math.sqrt(a**2 + b**2)

def area(a,b):
	"""Cálcula el area dados dos lados"""
	return (a*b)/2
