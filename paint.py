from PIL import Image, ImageDraw
import math
import random
import numpy
import torch
import util

def generate_grid_image(n: int, m: int, shift=[0,0], save_path: str = 'grid.png', dft: bool = False):
    img_size=n
    n=n//m+1
    img = Image.new('RGB', (img_size, img_size), color='white')
    draw = ImageDraw.Draw(img)

    for row in range(-1,n):
        for col in range(-1,n):
            color = 'black' if (row + col) % 2 == 0 else 'white'
            if col* m >= img_size or row * m >= img_size:
                continue
            top_left = (col * m+shift[0], row * m+shift[1])
            bottom_right = (top_left[0] + m, top_left[1] + m)
            draw.rectangle([top_left, bottom_right], fill=color)

    if dft:
        img=numpy.array(img)
        img=torch.tensor(img).permute(2,0,1)/255
        print(img.max())
        for i in range(3):
            img[i]=util.dft(img[i])
        img=img/img.max()
        img=(img*255).permute(1,2,0).numpy().astype('uint8')
        img=Image.fromarray(img)
    img.save(save_path)
    print(f"Save to {save_path}")

if __name__ == "__main__":
    n= 256
    m= 16
    generate_grid_image(n=n, m=m, save_path=f'grid_{n}_{m}.png', dft=True)
    for m in range(16,32):
        for id in range(16):
            generate_grid_image(n=n, m=m, shift=[random.randint(0,m-1),random.randint(0,m-1)], save_path=f'data/train/class0/grid_{n}_{m}_{id}.png')
