#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def to_red(image):
	red_img = image.copy()
	red_img[..., 1] = 0
	red_img[..., 2] = 0
	return red_img

def to_blue(image):
	blue_img = image.copy()
	blue_img[..., 0] = 0
	blue_img[..., 1] = 0
	return blue_img

def to_green(image):
	green_img = image.copy()
	green_img[..., 0] = 0
	green_img[..., 2] = 0
	return green_img

def to_grayscale(image):
	gray_weights = np.array([0.2126, 0.7152, 0.0722])
	gray_img = np.dot(image, gray_weights)
	return gray_img

def main():
	image = plt.imread("src/painting.png")
	gray_image = to_grayscale(image)
	plt.imshow(gray_image)
	plt.gray()
	plt.show()

	fig, axs = plt.subplots(3, 1, figsize=(6, 9))

	axs[0].imshow(to_red(image))
	axs[0].set_title("Canal Rojo")
	axs[0].axis("off")

	axs[1].imshow(to_green(image))
	axs[1].set_title("Canal Verde")
	axs[1].axis("off")

	axs[2].imshow(to_blue(image))
	axs[2].set_title("Canal Azul")
	axs[2].axis("off")

	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
    main()
