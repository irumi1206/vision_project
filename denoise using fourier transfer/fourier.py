import cv2
import matplotlib.pyplot as plt
import numpy as np

##### To-do #####
def custom_fft2(img):
    fft2_img=np.zeros(shape=(img.shape[0],img.shape[1]),dtype=np.complex128)
    for y in range(0,img.shape[0]):
        for x in range(0,img.shape[1]):
            print(y,x)
            for i in range(0,img.shape[0]):
                for j in range(0,img.shape[1]):
                    fft2_img[y][x] += img[i][j] * np.exp(-1j*2*(np.pi) * (y*i/(img.shape[0]) + x*j/(img.shape[1])))
    return fft2_img
def custom_ifft2(img):
    ifft2_img=np.zeros(shape=(img.shape[0],img.shape[1]),dtype=np.complex128)
    for y in range(0,img.shape[0]):
        for x in range(0,img.shape[1]):
            print(y,x)
            for i in range(0,img.shape[0]):
                for j in range(0,img.shape[1]):
                    ifft2_img[y][x]+=img[i][j]+np.exp(1j*2*(np.pi)*(y*i/(img.shape[0])+x*j/(img.shape[1])))
    ifft2_img=ifft2_img/((img.shape[0])*(img.shape[1]))
    return ifft2_img

def centering(img):
    height=img.shape[0]
    width=img.shape[1]
    mid_height=int(height/2)
    mid_width=int(width/2)
    centered_img=img.copy()
    for y in range(0,height):
        for x in range(0,width):
            new_y=y
            new_x=x

            if new_y>=mid_height:
                new_y-=mid_height
            else:
                new_y+=mid_height

            if new_x>=mid_width:
                new_x-=mid_width
            else:
                new_x+=mid_width
            centered_img[new_y][new_x]=img[y][x]
    
    return centered_img

def fm_spectrum(img):
    img_fourier=np.fft.fft2(img)
    img_fourier_shift=centering(img_fourier)
    spectrum_img=np.log(2+np.abs(img_fourier_shift))  
    
    return spectrum_img

def low_pass_filter(img, th=20):
    img_fourier=np.fft.fft2(img)
    img_fourier_shift=centering(img_fourier)
    
    height=img.shape[0]
    mid_height=int(height/2)
    width=img.shape[1]
    mid_width=int(width/2)
    low_pass_filter=np.zeros((height,width),dtype=np.complex128)
    for y in range(low_pass_filter.shape[0]):
        for x in range(low_pass_filter.shape[1]):
            if (y-mid_height)**2+(x-mid_width)**2<=th*th:
                low_pass_filter[y][x]=1
    
    low_pass_fourier=img_fourier_shift*low_pass_filter
    low_pass_img=np.fft.ifft2(centering(low_pass_fourier))
    
    return low_pass_img.real

def high_pass_filter(img, th=30):
    img_fourier=np.fft.fft2(img)
    img_fourier_shift=centering(img_fourier)
    
    height=img.shape[0]
    mid_height=int(height/2)
    width=img.shape[1]
    mid_width=int(width/2)
    high_pass_filter=np.ones((height,width),dtype=np.complex128)
    for y in range(high_pass_filter.shape[0]):
        for x in range(high_pass_filter.shape[1]):
            if (y-mid_height)**2+(x-mid_width)**2<=th*th:
                high_pass_filter[y][x]=0
    high_pass_fourier=img_fourier_shift*high_pass_filter
    high_pass_img=np.fft.ifft2(centering(high_pass_fourier))
    
    return high_pass_img.real

def denoise1(img):
    img_fourier=np.fft.fft2(img)
    img_fourier_shift=centering(img_fourier)
    coors=[(11,11,0),(11,81,0),(11,192,0),(11,298,0),(11,408,0),(11,491,0),
    (33,37,0),(33,134,0),(33,354,0),(33,461,0),
    (85,11,0),(85,81,0),(85,192,0),(85,298,0),(85,408,0),(85,491,0),
    (139,37,0),(139,134,0),(139,354,0),(139,461,0),
    (195,11,0),(195,81,0),(195,192,0),(195,298,0),(195,408,0),(195,491,0),
    (303,11,0),(303,81,0),(303,192,0),(303,298,0),(303,408,0),(303,491,0),
    (357,37,0),(357,134,0),(357,354,0),(357,461,0),
    (414,11,0),(414,81,0),(414,192,0),(414,298,0),(414,408,0),(414,491,0),
    (466,37,0),(466,134,0),(466,354,0),(466,461,0),
    (490,11,0),(490,81,0),(490,192,0),(490,298,0),(490,408,0),(490,491,0)
    ]

    for coor in coors:
        for y in range(coor[0],coor[0]+14):
            for x in range(coor[1],coor[1]+20):
                img_fourier_shift[y][x]=coor[2]

    clean_img=np.fft.ifft2(centering(img_fourier_shift))

    return clean_img.real

def denoise2(img):
    img_fourier=np.fft.fft2(img)
    img_fourier_shift=centering(img_fourier)
    mid_height=int(img_fourier.shape[0]/2)
    mid_width=int(img_fourier_shift.shape[1]/2)
    for y in range(0,img_fourier.shape[0]):
        for x in range(0,img_fourier.shape[1]):
            if (y-mid_height)**2+(x-mid_width)**2<=45*45 and (y-mid_height)**2+(x-mid_width)**2>=40*40:
                img_fourier_shift[y][x]=0


    clean_img=np.fft.ifft2(centering(img_fourier_shift))

    return clean_img.real

#################

if __name__ == '__main__':
    img = cv2.imread('task2_sample.png', cv2.IMREAD_GRAYSCALE)
    cor1 = cv2.imread('task2_corrupted_1.png', cv2.IMREAD_GRAYSCALE)
    cor2 = cv2.imread('task2_corrupted_2.png', cv2.IMREAD_GRAYSCALE)

    cv2.imwrite('denoised1.png',denoise1(cor1))
    cv2.imwrite('denoised2.png',denoise2(cor2))

    def drawFigure(loc, img, label):
        plt.subplot(*loc), plt.imshow(img, cmap='gray')
        plt.title(label), plt.xticks([]), plt.yticks([])
    drawFigure((2,7,1), img, 'Original')
    drawFigure((2,7,2), low_pass_filter(img), 'Low-pass')
    drawFigure((2,7,3), high_pass_filter(img), 'High-pass')
    drawFigure((2,7,4), cor1, 'Noised')
    drawFigure((2,7,5), denoise1(cor1), 'Denoised')
    drawFigure((2,7,6), cor2, 'Noised')
    drawFigure((2,7,7), denoise2(cor2), 'Denoised')

    drawFigure((2,7,8), fm_spectrum(img), 'Spectrum')
    drawFigure((2,7,9), fm_spectrum(low_pass_filter(img)), 'Spectrum')
    drawFigure((2,7,10), fm_spectrum(high_pass_filter(img)), 'Spectrum')
    drawFigure((2,7,11), fm_spectrum(cor1), 'Spectrum')
    drawFigure((2,7,12), fm_spectrum(denoise1(cor1)), 'Spectrum')
    drawFigure((2,7,13), fm_spectrum(cor2), 'Spectrum')
    drawFigure((2,7,14), fm_spectrum(denoise2(cor2)), 'Spectrum')

    plt.show()