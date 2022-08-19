import cv2
import imgaug.augmenters as iaa
import os

images = []
for image in os.listdir('/image'):
  if image.endswith('.jpg'):
    print('/image/' + image)
    img = cv2.imread('/image/' + image)
    images.append(img)


aug = iaa.Sequential([
  iaa.Rotate((-50,50)),
  iaa.Affine(shear=(-25,25)),
  iaa.Crop(percent=(0,0.1)),
  iaa.LinearContrast((0.75,1.5)),
  iaa.Fliplr(0.5)
], random_order = True)

aug_image = aug(images=images) 

count = 0
for x in aug_image:
  cv2.imwrite(str(count) + ".jpgnew" + '.jpg',x)
  count += 1

