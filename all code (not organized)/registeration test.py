import numpy as np
import SimpleITK as sitk
from skimage import io
import cv2

img1 = io.imread("C:/Users/JiangQin/Documents/python/Music Composition Project/code files/test1.png")
img2 = io.imread("C:/Users/JiangQin/Documents/python/Music Composition Project/code files/test2.png")
img1 = img1[:,:,0]
img2 = img2[:,:,0]
print(np.max(img1),img1.dtype)

img1 = img1.astype("float32")
img2 = img2.astype("float32")

img1= img1/np.max(img1)
img2= img2/np.max(img2)

uno = sitk.GetImageFromArray(img1)
dos = sitk.GetImageFromArray(img2)

def register(fixed_image, moving_image, orig, transform = None):
    if transform is None:
        resamples = []
        metrics = []
        transforms = []
        for i in range (1,10):
            ImageSamplingPercentage = 1
            initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.ScaleTransform(2,(1, 0)), sitk.CenteredTransformInitializerFilter.MOMENTS)
            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=200)
            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(float(ImageSamplingPercentage)/100)
            registration_method.SetInterpolator(sitk.sitkLinear)
            registration_method.SetOptimizerAsGradientDescent(learningRate=0.001, numberOfIterations=10**5, convergenceMinimumValue=1e-6, convergenceWindowSize=100) #Once
            registration_method.SetOptimizerScalesFromPhysicalShift() 
            registration_method.SetInitialTransform(initial_transform)

            transform = registration_method.Execute(fixed_image, moving_image)
            #print(transform)
            print("number:",i)
            print(registration_method.GetMetricValue())
            metrics.append(registration_method.GetMetricValue())
            resamples.append(sitk.Resample(orig, fixed_image, transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID()))
            transforms.append(transform)
        print(np.min(metrics))
        return sitk.GetArrayFromImage(resamples[metrics.index(np.min(metrics))]),transforms[metrics.index(np.min(metrics))]
    else:
        return sitk.GetArrayFromImage(sitk.Resample(orig, fixed_image, transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())),transform

blub,_ = register(uno,dos,dos)


cv2.imwrite("C:/Users/JiangQin/Documents/python/Music Composition Project/code files/testoutput.png", blub*255)



