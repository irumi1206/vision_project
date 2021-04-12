import cv2
import utils
import numpy as np

def task1(src_img_path, clean_img_path, dst_img_path):
    """
    This is main function for task 1.
    It takes 3 arguments,
    'src_img_path' is path for source image.
    'clean_img_path' is path for clean image.
    'dst_img_path' is path for output image, where your result image should be saved.

    You should load image in 'src_img_path', and then perform task 1 of your assignment 1,
    and then save your result image to 'dst_img_path'.
    """
    noisy_img = cv2.imread(src_img_path)
    clean_img = cv2.imread(clean_img_path)
    result_img = None

    # do noise removal
    result_img=apply_median_filter(noisy_img,3)

    new_image=apply_bilateral_filter(noisy_img,5,10,70)
    if utils.calculate_rms_cropped(clean_img,result_img)> utils.calculate_rms_cropped(clean_img,new_image):
        result_img=new_image

    new_image=apply_bilateral_filter(noisy_img,5,10,30)
    if utils.calculate_rms_cropped(clean_img,result_img)> utils.calculate_rms_cropped(clean_img,new_image):
        result_img=new_image

    cv2.imwrite(dst_img_path, result_img)
    pass


def apply_average_filter(img, kernel_size):
    """
    You should implement average filter convolution algorithm in this function.
    It takes 2 arguments,
    'img' is source image, and you should perform convolution with average filter.
    'kernel_size' is a int value, which determines kernel size of average filter.

    You should return result image.
    """

    length=int((kernel_size-1)/2)
    #make image with padding which is filled with 0
    img_with_padding=np.zeros(shape=(img.shape[0]+kernel_size-1,img.shape[1]+kernel_size-1,img.shape[2]))
    for x in range(0,img.shape[0]):
        for y in range(0,img.shape[1]):
            for rgb in range(0,img.shape[2]):
                img_with_padding[x+length][y+length][rgb]=img[x][y][rgb]
    #make average filter
    average_filter=(np.ones(shape=(kernel_size,kernel_size))/(kernel_size*kernel_size))
    #multiply average filter
    filtered_img=np.zeros(shape=(img.shape[0],img.shape[1],img.shape[2]))
    for x in range(0,img.shape[0]):
        for y in range(0,img.shape[1]):
            for rgb in range(0,img.shape[2]):
                img_slice=img_with_padding[x:x+2*length+1,y:y+2*length+1,rgb]
                filtered_img[x][y][rgb]=int((img_slice*average_filter).sum())

    return filtered_img


def apply_median_filter(img, kernel_size):
    """
    You should implement median filter convolution algorithm in this function.
    It takes 2 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is a int value, which determines kernel size of median filter.

    You should return result image.
    """

    length=int((kernel_size-1)/2)
    #make image with padding which is filled with 0
    img_with_padding=np.zeros(shape=(img.shape[0]+kernel_size-1,img.shape[1]+kernel_size-1,img.shape[2]))
    for x in range(0,img.shape[0]):
        for y in range(0,img.shape[1]):
            for rgb in range(0,img.shape[2]):
                img_with_padding[x+length][y+length][rgb]=img[x][y][rgb]
    filtered_img=np.zeros(shape=(img.shape[0],img.shape[1],img.shape[2]))
    for x in range(0,img.shape[0]):
        for y in range(0,img.shape[1]):
            for rgb in range(0,img.shape[2]):
                img_slice=img_with_padding[x:x+2*length+1,y:y+2*length+1,rgb]
                filtered_img[x][y][rgb]=int(np.median(img_slice))

    return filtered_img


def apply_bilateral_filter(img, kernel_size, sigma_s, sigma_r):
    """
    You should implement convolution with additional filter.
    You can use any filters for this function, except average, median filter.
    It takes at least 2 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is a int value, which determines kernel size of average filter.
    'sigma_s' is a int value, which is a sigma value for G_s
    'sigma_r' is a int value, which is a sigma value for G_r

    You can add more arguments for this function if you need.

    You should return result image.
    """

    length=int((kernel_size-1)/2)
    #image with padding is created
    img_with_padding=np.zeros(shape=(img.shape[0]+kernel_size-1,img.shape[1]+kernel_size-1,img.shape[2]))
    for x in range(0,img.shape[0]):
        for y in range(0,img.shape[1]):
            for rgb in range(0,img.shape[2]):
                img_with_padding[x+length][y+length][rgb]=img[x][y][rgb]
    #make space gaussian filter which is same for every pixel, constant multiplication is ignored because it will be normalized later
    gaussian_space_filter=np.zeros(shape=(kernel_size,kernel_size))
    for x in range(0,kernel_size):
        for y in range(0,kernel_size):
            gaussian_space_filter[x][y]=np.exp(-((x-length)*(x-length)+(y-length)*(y-length))/(2*sigma_s*sigma_s))
    filtered_img=np.zeros(shape=(img.shape[0],img.shape[1],img.shape[2]))
    for x in range(0,img.shape[0]):
        for y in range(0,img.shape[1]):
            for rgb in range(0,img.shape[2]):
                img_slice=img_with_padding[x:x+2*length+1,y:y+2*length+1,rgb]
                gaussian_range_filter=np.ones(shape=(kernel_size,kernel_size))
                for x1 in range(0,kernel_size):
                    for y1 in range(0,kernel_size):
                        #range gaussian filter, constant multiplication is ignored because it will be normalized later
                        gaussian_range_filter[x1][y1]=np.exp(-((img_with_padding[x+length][y+length][rgb]-img_with_padding[x+x1][y+y1][rgb])**2)/(2*sigma_r*sigma_r))
                bilateral_filter=gaussian_space_filter*gaussian_range_filter
                #after summing up, it is normalized
                filtered_img[x][y][rgb]=int(((img_slice*bilateral_filter).sum())/(bilateral_filter.sum()))

    return filtered_img
