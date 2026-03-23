import imageio.v3 as imageio
import re

image = imageio.imread('pompom_segredo.bmp')

channels = [image[:, :, i] for i in range(3)]

chars = list(map(lambda channel: ''.join([chr(value) for value in channel.flatten(order="C")]), channels))
chars2 = list(map(lambda channel: ''.join([chr(value) for value in channel.flatten(order="F")]), channels))
chars.extend(chars2)

for color in chars:
    matches = re.finditer(r'segredo:[^;]*;', color)
    for match in matches:
        print(match.group()[9:-1])