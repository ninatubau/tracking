import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from tifffile import imread, imsave

def main():
	global x_point, y_point, image
	image = imread('result.tif')
	print(image.shape)
	df = pd.read_csv('/home/ninatubau/Desktop/dataframe.csv')

	#for i in range(30):
	point = df.iloc[[9]].as_matrix()
	print(point.shape)
	even = np.arange(0,258,4)
	odd = np.arange(1,258,4)
	x_point = []
	y_point = []

	for i in even: 
	    x_point.append(point[0,i]) 
	for i in odd: 
	    y_point.append(point[0,i])   

	fig = plt.figure(figsize=(20,20))
	ax = fig.add_subplot(121)
	ax.set_xlim([0, 661])
	ax.set_ylim([0, 659])
	plt.gca().invert_yaxis()
	scat = plt.scatter(x_point[0],y_point[0],  cmap="jet", edgecolor="k") 
	scat.set_alpha(0.8)
	ani = animation.FuncAnimation(fig, updatefig, frames=129, fargs=(fig, scat), interval=300)
	#ani = animation.FuncAnimation(fig, updatefig, frames=100, interval=100)
	

	ax = fig.add_subplot(122)
	first_im = convert_rgb(image[0,:,:,:])
	im = plt.imshow(first_im) 
	ani2 = animation.FuncAnimation(fig, updatefig2, frames=129, fargs=(fig, im), interval=300)
	#ani = animation.FuncAnimation(fig, updatefig, frames=100, interval=100)
	plt.show()

def convert_rgb(im_):
	im_ = 0.3*im_[:,:,0]+0.59*im_[:,:,1]+0.11*im_[:,:,2]
	return im_


def updatefig(i, fig, scat):
	print(np.array([[x_point[i],y_point[i]]]))
	scat.set_offsets(np.array([[x_point[i],y_point[i]]]))
	return scat,

def updatefig2(i, fig, im):
	im.set_array(convert_rgb(image[i]))
	return im,

main()