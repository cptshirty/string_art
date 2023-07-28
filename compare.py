import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as Img


img_real = Img.open("square.png")
img_fake = Img.open("square_string.png")
img_real = img_real.convert('L')
img_fake = img_fake.convert('L')
img_arr = 1 - np.array(img_real)/255
fake_arr = np.array(img_fake)/255

plt.imshow(img_arr - fake_arr)
plt.show()

