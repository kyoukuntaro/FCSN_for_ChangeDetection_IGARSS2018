import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

IMAGE_SIZE = 112
IMAGE_NUMBER = 2000
VISUALIZE = 0
MAIN_DIR = './'
IMAGE_FOLDER = 'images'
LABEL_FOLDER = 'labels'

def main():
    main_dir = MAIN_DIR
    images_dir = main_dir + IMAGE_FOLDER
    labels_dir = main_dir + LABEL_FOLDER
    city_nm_ls = os.listdir(labels_dir)
    city_nm_ls.remove('README.txt')
    if not('data' in os.listdir(main_dir)):
        os.mkdir(main_dir+'data')
    for city_nm in city_nm_ls:
        img_file1 = images_dir + city_nm + '/pair/img1.png'
        img1 = Image.open(img_file1)
        img1 = np.asarray(img1)
        img_file2 = images_dir + city_nm + '/pair/img2.png'
        img2 = Image.open(img_file2)
        img2 = np.asarray(img2)
        lbl_file = labels_dir + city_nm + '/cm/cm.png'
        lbl = Image.open(lbl_file)
        lbl = np.asarray(lbl)
        for i in range(int((IMAGE_NUMBER-1)/len(city_nm_ls))+1):
            x = np.random.randint(img1.shape[0]-IMAGE_SIZE)
            y = np.random.randint(img1.shape[1]-IMAGE_SIZE)
            mini_img1 = img1[x:x+IMAGE_SIZE,y:y+IMAGE_SIZE,:]
            mini_img2 = img2[x:x+IMAGE_SIZE,y:y+IMAGE_SIZE,:]
            mini_lbl = lbl[x:x+IMAGE_SIZE,y:y+IMAGE_SIZE]
            data = np.zeros([7,IMAGE_SIZE,IMAGE_SIZE])
            data[:3,:,:] = mini_img1.transpose([2,0,1])
            data[3:6,:,:] = mini_img2.transpose([2,0,1])
            data[6,:,:] = mini_lbl[:,:,0]
            file_nm = main_dir + 'data/' + city_nm + str(i).zfill(3) + '.npy'
            np.save(file_nm, data)
            if VISUALIZE:
                plt.figure(figsize=(20,30))
                plt.subplot(1,3,1)
                plt.imshow(mini_img1)
                plt.subplot(1,3,2)
                plt.imshow(mini_img2)
                plt.subplot(1,3,3)
                plt.imshow(mini_lbl[:,:,0])
                plt.show()

if __name__=='__main__':
    main
