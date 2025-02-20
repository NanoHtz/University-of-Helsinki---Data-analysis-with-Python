#!/usr/bin/env python3
import math

def triangulo():
    base = float(input("Give base of the triangle: "))  # Cambié los mensajes para coincidir con el test
    altura = float(input("Give height of the triangle: "))
    return (base * altura) / 2  # Retornamos el área en lugar de imprimir

def rectangulo():
    base = float(input("Give width of the rectangle: "))
    altura = float(input("Give height of the rectangle: "))
    return base * altura

def circulo():
    radius = float(input("Give radius of the circle: "))
    return (radius**2) * math.pi

def shape():
    while True:
        shape = input("Choose a shape (triangle, rectangle, circle): ").strip().lower()

        if shape == "":
            break  # Sale del bucle si la entrada está vacía

        elif shape == "triangle":
            area = triangulo()
            print(f"The area is {area:.6f}")  # Ahora imprimimos el mensaje EXACTO esperado

        elif shape == "rectangle":
            area = rectangulo()
            print(f"The area is {area:.6f}")

        elif shape == "circle":
            area = circulo()
            print(f"The area is {area:.6f}")

        else:
            print("Unknown shape!")

def main():
    shape()

if __name__ == "__main__":
    main()
