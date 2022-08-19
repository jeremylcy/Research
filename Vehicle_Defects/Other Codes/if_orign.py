import os
from skimage import io
import numpy as np

print('\n#####################################################')
print('                       WELCOME!                      ')
print('This is to help you find and remove duplicated images')
print('               from a designated folder              ')
print('#####################################################')
folder = input('\nProvide your images folder path :\n')
exists = os.path.isdir(folder)
if exists == True:
    path = folder
else:
    print('invalid path\n')
    exit()
    
def main():
    
    # path = os.path.join(os.getcwd(), 'test')
    files = os.listdir(path)

    imgs = []
    print('Detecting total images at destination')
    for index, file in enumerate(files):
        if file.endswith('.png') or file.endswith('.jpg'):
            new_path = os.path.join(path, file)
            imgs.append(io.imread(new_path))

    
    print('\nDetected images in folder : ' + str(index + 1))
    print('\n>>>  Checking and removing any duplicated images  <<<')
    print('>>>                  Please wait                  <<<\n')
    final = []
    dup = []
    for j in range(0, len(imgs)):
        if j < len(imgs):
            
            image1 = imgs[j]
            for i in range((j + 1), len(imgs)):
                if i < len(imgs):
                    image2 = imgs[i]

                    # if same, check if data is same
                    data = np.array_equal(image1, image2)
                    dup.append(data)
                    if data == True:
                        print('found duplicate')
                        imgs.pop(i)
            
            print('In progress')
            final.append(imgs[j])

    s = 0
    for i in range(0, len(dup)):
        if dup[i] == True:
            s += 1

    for index, file in enumerate(files):
        if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jfif') or file.endswith('.jpeg'):
            print('delete current images')
            new_path = os.path.join(path, file)
            os.remove(new_path)

    for i in range(0, len(final)):
        print('re-fitting images')
        final_path = os.path.join(path, str(i) + '.png')
        io.imsave(final_path, final[i])

    if s != 0:
        print('\nDetected several duplicated images not cleared')
        print(' Re-running system again, please wait, thanks ')
        main()
    else:
        print('\nSuccess, there are no more duplicated images detected')
        print('#####################################################\n')
        exit()

#how_many_times = input('\nDo you want to continue with action? (y/n): ')
main()