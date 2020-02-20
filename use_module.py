from feature_extractor_class import feature_extraction
from PIL import Image
import time
from io import StringIO
from numpy import asarray

name = 'ft_net_dense'
print("----------------Using model ", name, "----------------")

t_start_load = time.time_ns()
image0 = Image.open("../testdir/WIN_20200218_11_23_35_Pro (2).jpg")
image1 = Image.open("../testdir/0003_c1s6_015971_02.jpg")
image2 = Image.open("../testdir/WIN_20200218_11_23_35_Pro.jpg")
image3 = Image.open("../testdir/WIN_20200218_11_23_36_Pro.jpg")
image4 = Image.open("../testdir/WIN_20200218_12_10_56_Pro.jpg")
image5 = Image.open("../testdir/WIN_20200218_12_11_00_Pro.jpg")
image0.load()
image1.load()
image2.load()
image3.load()
image4.load()
image5.load()
t_end_load = time.time_ns()
print("Image loading took: ", (t_end_load-t_start_load)/1000000000, " seconds")
imgArray = [image0, image1, image2, image3, image4]
imgArray += imgArray

t_start = time.time_ns()
fe = feature_extraction('0', name, '1', 10)
t_model_loaded = time.time_ns()

print("Model loading time is: ", (t_model_loaded - t_start)/1000000000, 'seconds')

t_feature_start = time.time_ns()
print("Processing",len(imgArray), "pictures:")
feature2 = fe.extract_feature(imgArray)
#print("f1: ", feature1, "ft2: ", feature2)
t_feature_end = time.time_ns()
#print("Feature: ", feature)
print("Extracted feature from ", len(imgArray), " images in ", (t_feature_end-t_feature_start)/1000000000, 'seconds')