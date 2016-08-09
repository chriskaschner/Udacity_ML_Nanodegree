from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convolve(image, kernel):
    # get dimensions of input image and kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    pad = (kW - 1)/2
    image = cv2.copyMakeBorder(image,pad, pad, pad, pad,
                               cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

    # loop over input image, "sliding" the kernel along
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            #extract ROI (region of interest) of image
            # by extracting the center region of the current
            # coordinates
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # perform convolution
            k = (roi * kernel).sum()

            # store convolved values in the output
            output[y - pad, x - pad] = k

    #rescale output image
    output = rescale_intensity(output, in_range=(0,255))
    output = (output * 255).astype("uint8")

    # return output image
    return output

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
               help="path to the input image")
args = vars(ap.parse_args())

# smoothing kernels
smallBlur = np.ones((7,7), dtype="float") * (1.0 / (7*7))
largeBlur = np.ones((21,21), dtype="float") * (1.0 / (21 * 21))

#sharpening filter
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

kernelBank = (
    ("small_blur", smallBlur),
    ("large_blur", largeBlur),
    ("sharpen", sharpen))

# load image, convert to greyscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# loop over kernels
for (kernelName, kernel) in kernelBank:
    # apply kernel to greyscale image
    # using both our convolve function
    # & OpenCV's "filter2d"
    print("[INFO] applying {} kernel".format(kernelName))
    convolveOutout= convolve(gray, kernel)
    opencvOutput= cv2.filter2D(gray, -1, kernel)

    #show output images
    cv2.imshow("original", gray)
    cv2.imshow("{} - convolve".format(kernelName), convolveOutout)
    cv2.imshow("{} - openCV".format(kernelName), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows
