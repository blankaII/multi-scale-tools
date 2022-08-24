import sys, os #<----
import openslide #<----
from PIL import Image #<----
import numpy as np #<----
### import pandas as pd
### from collections import Counter
### from matplotlib import pyplot as plt
from skimage import io #<----
import threading #<----
import time #<----
### import collections
### import cv2
### import albumentations as A
### import time
from skimage import exposure
### import json
import utils
import argparse

np.random.seed(0)

parser = argparse.ArgumentParser(description='Configurations of the parameters for the extraction')
parser.add_argument('-m', '--MAGS', help='wanted magnification to extract the patches(色塊)',nargs="+", type=int, default=[10,5])
parser.add_argument('-w', '--MASK_LEVEL', help='wanted magnification to extract the patches(色塊)',type=float, default=1.25)
parser.add_argument('-i', '--INPUT_DATA', help='input data: it can be a .csv file with the paths or a directory',type=str, default='/home/niccolo/ExamodePipeline/Multiple_Instance_Learning/Colon/csv_folder/partitions.csv')
parser.add_argument('-t', '--INPUT_MASKS', help='directory where the masks are stored',type=str, default='/home/niccolo/ExamodePipeline/Multiple_Instance_Learning/Colon/MASKS/')
parser.add_argument('-o', '--PATH_OUTPUT', help='directory where the patches will be stored',type=str, default='/home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/')
parser.add_argument('-p', '--THREADS', help='amount of threads to use',type=int, default=10)
### 其實ROI沒用，因為find_border_coordinates裡面只寫了ROI==False的condition
parser.add_argument('-r', '--ROI', help='if the WSI is composed of similar slices, only one of them is used (True) or all of them (False)',type=bool, default=False)
parser.add_argument('-s', '--SIZE', help='patch size',type=int, default=224)
parser.add_argument('-x', '--THRESHOLD', help='threshold of tissue pixels to select a patches',type=float, default=0.7)
### STRIDE根本沒用過
parser.add_argument('-y', '--STRIDE', help='pixel_stride between patches',type=int, default=0)

args = parser.parse_args()

### wanted magnification to extract the patches
MAGNIFICATIONS = args.MAGS
### [10,5]
MAGNIFICATIONS_str = str(MAGNIFICATIONS)
### "[10,5]"
MAGNIFICATIONS_str = MAGNIFICATIONS_str.replace(" ", "")
### "[10,5]"
MAGNIFICATIONS_str = MAGNIFICATIONS_str.replace(",", "_")
### "[10_5]"
MAGNIFICATIONS_str = MAGNIFICATIONS_str.replace("[", "_")
### "_10_5]"
MAGNIFICATIONS_str = MAGNIFICATIONS_str.replace("]", "_")
### MAGNIFICATIONS_str="_10_5_"
### wanted magnification to extract the patches

### MASK_LEVEL=args.MASK_LEVEL=1.25: wanted magnification to extract the patches
MASK_LEVEL = args.MASK_LEVEL
### 1.25

### input data: it can be a .csv file with the paths or a directory
LIST_FILE = args.INPUT_DATA
### /home/niccolo/ExamodePipeline/Multiple_Instance_Learning/Colon/csv_folder/partitions.csv

### directory where the masks are stored
PATH_INPUT_MASKS = args.INPUT_MASKS
### /home/niccolo/ExamodePipeline/Multiple_Instance_Learning/Colon/MASKS/

### directory where the patches will be stored
PATH_OUTPUT = args.PATH_OUTPUT
### /home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/
utils.create_dir(PATH_OUTPUT)
PATH_OUTPUT = PATH_OUTPUT+'multi_magnifications_centers/'
### /home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/multi_magnifications_centers/
utils.create_dir(PATH_OUTPUT)
### PATH_OUT: directory where the patches will be stored + MAGNIFICATIONS_ + wanted magnification to extract the patches
PATH_OUTPUT = PATH_OUTPUT+'MAGNIFICATIONS_'+MAGNIFICATIONS_str+'/'
### /home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/multi_magnifications_centers/MAGNIFICATIONS__10_5_/
### PATH_OUT: directory where the patches will be stored + MAGNIFICATIONS_ + wanted magnification to extract the patches
utils.create_dir(PATH_OUTPUT)

### patch size
new_patch_size = args.SIZE
### 224

### amount of threads to use
THREAD_NUMBER = args.THREADS
### 10
lockList = threading.Lock()
lockGeneralFile = threading.Lock()

### 其實沒用，因為find_border_coordinates只寫了ROI == False
### if the WSI is composed of similar slices, only one of them is used (True) or all of them (False)
ROI = args.ROI
### False
### threshold of tissue pixels to select a patches
THRESHOLD = args.THRESHOLD
### 0.7

def generate_parameters(WANTED_LEVEL, mags, WINDOW_WANTED_LEVEL):
    print("generate_parameters is running")
    #SELECTED_LEVEL = int(float(wsi.properties['aperio.AppMag']))

    #SELECTED_LEVEL = select_nearest_magnification(WANTED_LEVEL,mags)
    SELECTED_LEVEL = mags[0]
    ### line 127: mags = utils.available_magnifications(mpp, level_downsamples)
    ### MAGNIFICATION_RATIO: utils.available_magnifications(mpp, level_downsamples)/[wanted magnification to extract the patches]
    ### MAGNIFICATION_RATIO: available_magnifications/[wanted magnification to extract the patches]
    ### MASK_LEVEL=args.MASK_LEVEL=1.25: wanted magnification to extract the patches
    MAGNIFICATION_RATIO = SELECTED_LEVEL/MASK_LEVEL

    GLIMPSE_SIZE_SELECTED_LEVEL = WINDOW_WANTED_LEVEL*SELECTED_LEVEL/WANTED_LEVEL
    GLIMPSE_SIZE_SELECTED_LEVEL = int(GLIMPSE_SIZE_SELECTED_LEVEL)

    GLIMPSE_SIZE_MASK = GLIMPSE_SIZE_SELECTED_LEVEL/MAGNIFICATION_RATIO
    GLIMPSE_SIZE_MASK = int(GLIMPSE_SIZE_MASK)

    STRIDE_SIZE_MASK = 0
    TILE_SIZE_MASK = GLIMPSE_SIZE_MASK+STRIDE_SIZE_MASK

    return GLIMPSE_SIZE_MASK, GLIMPSE_SIZE_SELECTED_LEVEL, MAGNIFICATION_RATIO


#estrae glimpse e salva metadati relativi al glimpse
def analyze_file(filename):
    print("analyze_file is running")
    global filename_list_general, labels_multiclass_general, labels_binary_general, csv_binary, csv_multiclass, MAGNIFICATION

    patches = []

    file = openslide.open_slide(filename)
    ### file是openslide的file handler
    ### https://openslide.org/docs/properties/
    ### openslide.mpp-x
    ###     Microns(微米) per pixel in the X dimension of level 0. May not be present or accurate.
    mpp = file.properties['openslide.mpp-x']
    ### mpp = '0.22656727915354463'

    ### http://pointborn.com/article/2020/7/21/857.html <==不錯
    ### https://www.796t.com/content/1549681590.html
    ### https://blog.csdn.net/hjxu2016/article/details/70211198
    ### downsamples就是level 0往第k level的縮放倍數
    ### level_downsamples = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0)
    ### 比如在level[0],w=512
    ### 那在level[1],w=512*(level[0]/level[1])=512*(1/2)=256
    ### level[2],w=512*(level[0]/level[2])=512*(1/4)=128
    ### 所以level_downsamples就是zoom out的倍率
    level_downsamples = file.level_downsamples
    ### mags: available_magnifications
    mags = utils.available_magnifications(mpp, level_downsamples)
    ### openslide.mpp-x
    ###     Microns(微米) per pixel in the X dimension of level 0. May not be present or accurate.
    ### 因為mpp=0.226 < 0.26
    ###     所以magnification = 40
    ### mags = 40/(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0)
    ### mags = [40.0, 20.0, 10.0, 5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625, 0.078125]
    ### else mpp >= 0.26
    ###     所以magnification = 20
    ### mags = 20/(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0)
    ### mags = [20.0, 10.0, 5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625, 0.078125, 0.0390625]

    ### MAGNIFICATIONS: wanted magnification to extract the patches
    ### [10,5]
    ### wanted_levels: wanted magnification to extract the patches
    ### wanted_levels = [10,5]
    wanted_levels = MAGNIFICATIONS
    level = 0
            #load file
    #file = openslide.OpenSlide(filename)

            #load mask
    ### fname = "partitions.csv"
    fname = os.path.split(filename)[-1]
            #check if exists
    ### fname_mask: directory where the masks are stored + filename + _mask_use.png
    ### PATH_INPUT_MASKS=args.INPUT_MASKS: directory where the masks are stored
    fname_mask = PATH_INPUT_MASKS+fname+'/'+fname+'_mask_use.png'
    ### /home/niccolo/ExamodePipeline/Multiple_Instance_Learning/Colon/MASKS/partitions.csv/partitions.csv_mask_use.png

    ### array_dict後面沒用到
    array_dict = []

            #level 0 for the conversion
    ### wanted_levels=args.MAGS=MAGNIFICATIONS: wanted magnification to extract the patches
    ### MAGNIFICATIONS=[10,5]
    ### 所以wanted_levels=[10,5]
    ### WANTED_LEVEL=wanted_levels[0]=10: level 0 for the conversion(轉換)
    WANTED_LEVEL = wanted_levels[0]
    ### HIGHEST_LEVEL=mags[0]=40: available_magnifications的[0]
    HIGHEST_LEVEL = mags[0]
    #AVAILABLE_LEVEL = select_nearest_magnification(WANTED_LEVEL, mags, level_downsamples)

    ### WANTED_LEVEL=wanted_levels[0]=10: level 0 for the conversion
    ### MAGNIFICATIONS=[10,5]
    ### 所以wanted_levels=[10,5]
    ### MASK_LEVEL=args.MASK_LEVEL=1.25: wanted magnification to extract the patches(色塊)
    ### RATIO_WANTED_MASK= magnification/(wanted magnification to extract the patches)
    RATIO_WANTED_MASK = WANTED_LEVEL/MASK_LEVEL
    ### 10/1.25
    RATIO_HIGHEST_MASK = HIGHEST_LEVEL/MASK_LEVEL
    ### 40/1.25 = 32

    ### WINDOW_WANTED_LEVEL=new_patch_size=args.SIZE=224: patch size
    ### WINDOW_WANTED_LEVEL: patch size
    WINDOW_WANTED_LEVEL = new_patch_size
    ### 224

    ### glimpse: 快速瞥見, 一瞥
    ### WINDOW_WANTED_LEVEL=new_patch_size=args.SIZE=224: patch size
    ### GLIMPSE_SIZE_SELECTED_LEVEL=WINDOW_WANTED_LEVEL=args.SIZE=224: patch size
    ### GLIMPSE_SIZE_SELECTED_LEVEL: patch size=224
    GLIMPSE_SIZE_SELECTED_LEVEL = WINDOW_WANTED_LEVEL
    ### 224

    ### RATIO_WANTED_MASK = WANTED_LEVEL/MASK_LEVEL
    ### RATIO_WANTED_MASK= magnification/(wanted magnification to extract the patches)
    ### GLIMPSE_SIZE_SELECTED_LEVEL=WINDOW_WANTED_LEVEL=args.SIZE=224: patch size
    ### GLIMPSE_SIZE_MASK: WINDOW_WANTED_LEVEL/RATIO_WANTED_MASK
    ###                  : WINDOW_WANTED_LEVEL/(WANTED_LEVEL/MASK_LEVEL) : 224/(10/1.25)
    ###                  : 224/8
    GLIMPSE_SIZE_MASK = np.around(GLIMPSE_SIZE_SELECTED_LEVEL/RATIO_WANTED_MASK)
    ### 224/8 = 28
    GLIMPSE_SIZE_MASK = int(GLIMPSE_SIZE_MASK)
    ### 28
    ### 可以想成每一個patch的邊長

    GLIMPSE_HIGHEST_LEVEL = np.around(GLIMPSE_SIZE_MASK*RATIO_HIGHEST_MASK)
    ### 28*32=896
    GLIMPSE_HIGHEST_LEVEL = int(GLIMPSE_HIGHEST_LEVEL)
    ### 896

    STRIDE_SIZE_MASK = 0
    TILE_SIZE_MASK = GLIMPSE_SIZE_MASK+STRIDE_SIZE_MASK
    ### 28+0

    ### output_dir: directory where the patches will be stored + MAGNIFICATIONS_ + wanted magnification to extract the patches + fname
    output_dir = PATH_OUTPUT+fname
    ### /home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/multi_magnifications_centers/MAGNIFICATIONS__10_5_/partitions.csv
    #if (os.path.isfile(fname_mask) and os.path.isdir(output_dir)):
    ### fname_mask: directory where the masks are stored + filename + _mask_use.png
    ### fname_mask = /home/niccolo/ExamodePipeline/Multiple_Instance_Learning/Colon/MASKS/partitions.csv/partitions.csv_mask_use.png
    if (os.path.isfile(fname_mask)):
                    #creates directory
        ### output_dir: directory where the patches will be stored + MAGNIFICATIONS_ + wanted magnification to extract the patches + fname
        output_dir = PATH_OUTPUT+fname
        ### /home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/multi_magnifications_centers/MAGNIFICATIONS__10_5_/partitions.csv
        utils.create_dir(output_dir)
        ### /home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/multi_magnifications_centers/MAGNIFICATIONS__10_5_/partitions.csv/

        output_dir_m = []

        patches = []

        for m in MAGNIFICATIONS:
            ### MAGNIFICATIONS = [10,5]
            subdir_m = output_dir+'/magnification_'+str(m)+'x/'
            ### subdir_m = /home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/multi_magnifications_centers/MAGNIFICATIONS__10_5_/partitions.csv/magnification_10x/
            ### subdir_m = /home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/multi_magnifications_centers/MAGNIFICATIONS__10_5_/partitions.csv/magnification_5x/
            output_dir_m.append(subdir_m)
            ### ["/home/.../partitions.csv/magnification_10x","/home/.../partitions.csv/magnification_5x"]
            utils.create_dir(subdir_m)
            ### ["/home/.../partitions.csv/magnification_10x","/home/.../partitions.csv/magnification_5x"]

            #create CSV file structure (local)
        filename_list = [[] for m in MAGNIFICATIONS]
        ### filename_list = [[],[]]
        level_list = [[] for m in MAGNIFICATIONS]
        ### level_list = [[],[]]
        x_list = [[] for m in MAGNIFICATIONS]
        ### x_list = [[],[]]
        y_list = [[] for m in MAGNIFICATIONS]
        ### y_list = [[],[]]
        magnification_patches = [[] for m in MAGNIFICATIONS]
        ### magnification_patches = [[],[]]

        ### fname_mask: directory where the masks are stored + filename + _mask_use.png
        img = Image.open(fname_mask)
        ### PIL.Image.open("/home/niccolo/ExamodePipeline/Multiple_Instance_Learning/Colon/MASKS/partitions.csv/partitions.csv_mask_use.png")

        thumb = file.get_thumbnail(img.size)
        ### file從這邊來 ==> file = openslide.open_slide(filename)
        ### filename = file_from_par_csv.1
        thumb = thumb.resize(img.size)
        mask_np = np.asarray(thumb)
        img = np.asarray(img)

        mask_3d = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        ### >>> img
        ### array([['11', '12', '13'],
        ###        ['21', '22', '23']], dtype='<U2')
        ### >>> img[:,:]
        ### array([['11', '12', '13'],
        ###        ['21', '22', '23']], dtype='<U2')
        ### >>> img[:,:,np.newaxis]
        ### array([[['11'],
        ###         ['12'],
        ###         ['13']],
        ###
        ###        [['21'],
        ###         ['22'],
        ###         ['23']]], dtype='<U2')
        ### >>> np.repeat(img[:,:,np.newaxis],3)
        ### array(['11', '11', '11', '12', '12', '12', '13', '13', '13', '21', '21',
        ###        '21', '22', '22', '22', '23', '23', '23'], dtype='<U2')
        ### >>> np.repeat(img[:,:,np.newaxis],3,axis=2)
        ### array([[['11', '11', '11'],
        ###         ['12', '12', '12'],
        ###         ['13', '13', '13']],
        ###
        ###        [['21', '21', '21'],
        ###         ['22', '22', '22'],
        ###         ['23', '23', '23']]], dtype='<U2')

        ### mask_np從這邊來
        ### thumb = file.get_thumbnail(img.size)
        ### ### file從這邊來 ==> file = openslide.open_slide(filename)
        ### ### filename = file_from_par_csv.1
        ### thumb = thumb.resize(img.size)
        ### mask_np = np.asarray(thumb)

        WHITISH_THRESHOLD = utils.eval_whitish_threshold(mask_3d, mask_np)
        ### whitish 發白，接近白色的，灰白色的
        ### def eval_whitish_threshold(mask, thumb):
        ###     a = np.ma.array(thumb, mask=np.logical_not(mask))
        ###     mean_a = a.mean()
        ###
        ###     if (mean_a<=155):
        ###         THRESHOLD = 195.0
        ###     elif (mean_a>155 and mean_a<=180):
        ###         THRESHOLD = 200.0
        ###     elif (mean_a>180):
        ###         THRESHOLD = 205.0
        ###     return THRESHOLD

        mask_np = np.asarray(img)
        ### OOXX

        start_X, end_X, start_Y, end_Y = utils.find_border_coordinates(ROI, mask_np)
        ### ROI = False
        ### ROI => if the WSI is composed of similar slices, only one of them is used (True) or all of them (False)
        ### 其實沒用，因為find_border_coordinates只寫了ROI == False

        n_image = 0

        y_ini = start_Y + STRIDE_SIZE_MASK
        y_end = y_ini + GLIMPSE_SIZE_MASK

        while(y_end<end_Y):

            x_ini = start_X + STRIDE_SIZE_MASK
            x_end = x_ini + GLIMPSE_SIZE_MASK

            while(x_end<end_X):
                glimpse = mask_np[y_ini:y_ini+GLIMPSE_SIZE_MASK,x_ini:x_ini+GLIMPSE_SIZE_MASK]

                check_flag = utils.check_background_weakly(glimpse,THRESHOLD,TILE_SIZE_MASK)

                if(check_flag):

                    fname_patch = output_dir_m[0]+'/'+fname+'_'+str(n_image)+'.png'
                            #change to magnification 40x
                    center_x = x_ini+round(GLIMPSE_SIZE_MASK/2)
                    center_y = y_ini+round(GLIMPSE_SIZE_MASK/2)

                    x_coords_0 = int(x_ini*RATIO_HIGHEST_MASK)
                    y_coords_0 = int(y_ini*RATIO_HIGHEST_MASK)

                    patch_high = file.read_region((x_coords_0,y_coords_0),level,(GLIMPSE_HIGHEST_LEVEL,GLIMPSE_HIGHEST_LEVEL))
                    patch_high = patch_high.convert("RGB")

                    save_im = patch_high.resize((new_patch_size,new_patch_size))
                    save_im = np.asarray(save_im)

                    bool_white = utils.whitish_img(save_im,WHITISH_THRESHOLD)
                    bool_exposure = exposure.is_low_contrast(save_im)


                    if (bool_white):
                        if bool_exposure==False:
                        #if (exposure.is_low_contrast(save_im)==False):

                            io.imsave(fname_patch, save_im)

                            #add to arrays (local)
                            filename_list[0].append(fname_patch)
                            level_list[0].append(level)
                            x_list[0].append(x_coords_0)
                            y_list[0].append(y_coords_0)
                            magnification_patches[0].append(HIGHEST_LEVEL)

                            for a, m in enumerate(wanted_levels[1:], start = 1):
                                GLIMPSE_SIZE_MASK_LEVEL, GLIMPSE_SIZE_LEVEL, MAGNIFICATION_RATIO_LEVEL = generate_parameters(m, mags, WINDOW_WANTED_LEVEL)
                                y_ini_level = center_y - round(GLIMPSE_SIZE_MASK_LEVEL/2)
                                y_end_level = y_ini_level + GLIMPSE_SIZE_MASK_LEVEL

                                x_ini_level = center_x - round(GLIMPSE_SIZE_MASK_LEVEL/2)
                                x_end_level = x_ini_level + GLIMPSE_SIZE_MASK_LEVEL

                                x_coords_0 = int(x_ini_level*MAGNIFICATION_RATIO_LEVEL)
                                y_coords_0 = int(y_ini_level*MAGNIFICATION_RATIO_LEVEL)

                                patch_high = file.read_region((x_coords_0,y_coords_0),level,(GLIMPSE_SIZE_LEVEL,GLIMPSE_SIZE_LEVEL))
                                patch_high = patch_high.convert("RGB")
                                save_im = patch_high.resize((new_patch_size,new_patch_size))
                                save_im = np.asarray(save_im)

                                fname_patch = output_dir_m[a]+fname+'_'+str(n_image)+'.png'

                                io.imsave(fname_patch, save_im)
                                filename_list[a].append(fname_patch)
                                level_list[a].append(level)
                                x_list[a].append(x_coords_0)
                                y_list[a].append(y_coords_0)
                                magnification_patches[a].append(HIGHEST_LEVEL)

                            n_image = n_image+1
                            #save the image
                            #create_output_imgs(file_10x,fname)
                        else:
                            print("low_contrast " + str(output_dir))

                x_ini = x_end + STRIDE_SIZE_MASK
                x_end = x_ini + GLIMPSE_SIZE_MASK

            y_ini = y_end + STRIDE_SIZE_MASK
            y_end = y_ini + GLIMPSE_SIZE_MASK

            #add to general arrays
        if (n_image!=0):
            lockGeneralFile.acquire()
            filename_list_general.append(output_dir)

            print("len filename " + str(len(filename_list_general)) + "; WSI done: " + filename)
            print("extracted " + str(n_image) + " patches")
            lockGeneralFile.release()
            ### PATH_OUT: directory where the patches will be stored + MAGNIFICATIONS_ + wanted magnification to extract the patches
            utils.write_coords_local_file_CENTROIDS(PATH_OUTPUT,fname,[filename_list,level_list,x_list,y_list, magnification_patches],MAGNIFICATIONS)
            utils.write_paths_local_file_CENTROIDS(PATH_OUTPUT,fname,filename_list,MAGNIFICATIONS)
        else:
            print("ZERO OCCURRENCIES " + str(output_dir))

    else:
        print("no mask")

def explore_list(list_dirs):
    print("explore_list is running")
    global list_dicts, n, csv_binary, csv_multiclass
    #print(threadname + str(" started"))

    for i in range(len(list_dirs)):
        analyze_file(list_dirs[i])
        ### analyze_file(file_from_par_csv.1)
    #print(threadname + str(" finished"))

def main():
    print("main is running")
    #create output dir if not exists
    start_time = time.time()
    global list_dicts, n, filename_list_general, labels_multiclass_general, labels_binary_general, csv_binary, csv_multiclass


        #create CSV file structure (global)
    filename_list_general = []

    n = 0

    list_dirs = utils.get_input_file(LIST_FILE)
    ### LIST_FILE = /home/niccolo/ExamodePipeline/Multiple_Instance_Learning/Colon/csv_folder/partitions.csv
    ### list_dirs = partitions.csv 內的 file list

        #split in chunks for the threads
    list_dirs = list(utils.chunker_list(list_dirs,THREAD_NUMBER))
    ### list_dirs = ["file_from_par_csv.1","file_from_par_csv.2","file_from_par_csv.3"]
    print(len(list_dirs))

    threads = []
    for i in range(THREAD_NUMBER):
        t = threading.Thread(target=explore_list,args=([list_dirs[i]]))
        ### explore_list(list_dirs[0]) = explore_list("file_from_par_csv.1")
        ### explore_list(list_dirs[1]) = explore_list("file_from_par_csv.2")
        ### ...
        ### explore_list(list_dirs[9]) = explore_list("file_from_par_csv.3")
        threads.append(t)

    for t in threads:
        t.start()
        #time.sleep(60)

    for t in threads:
        t.join()

        #prepare data

    elapsed_time = time.time() - start_time
    print("elapsed time " + str(elapsed_time))


if __name__ == "__main__":
    main()
