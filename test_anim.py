import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from tifffile import imread, imsave
from matplotlib import cm

def main():
	global x_point, y_point, image
	global df
	global i_list,x_list
	i_list = []
	x_list = []
	image = imread('/data/data_drive/processing/Mike/data_median_threshold.tif')
	image = np.flip(image, axis=0)
	image_real = imread('/data/data_drive/processing/Mike/data.tif')

	df = pd.read_csv('dataframe_median.csv')
	df = df[df.columns[::-1]]

	#point = df.iloc[[2]].as_matrix()
	x_point = df[df.iloc[:,0] > -100].iloc[:, 0]
	y_point = df[df.iloc[:,1] > -100].iloc[:, 1]

	# print('shape of point',point.shape)
	# even = np.arange(0,512,4)
	# odd = np.arange(1,512,4)
	# x_point = []
	# y_point = []
	#
	# for i in even:
	#     x_point.append(point[0,i])
	# for j in odd:
	#     y_point.append(point[0,j])


	fig = plt.figure(figsize=(20,20))
	ax = fig.add_subplot(121)
	scat = ax.scatter(0, len(x_point))
	plt.xlim(0, 130)
	plt.ylim(0, 80)
	plt.title('Number of blobs per frame')
	plt.xlabel('Frame nb')
	plt.ylabel('Nb of blobs')
	ani = animation.FuncAnimation(fig, updatefig, frames=129, fargs=(fig, scat), interval=30)

	

	ax = fig.add_subplot(122)
	im = ax.imshow(image[0])
	N = len(x_point)
	print(N)
	rainb = cm.rainbow(np.linspace(0, 1, N))
	#theta = 2 * np.pi * np.random.rand(N)
	centroid = ax.scatter(x_point, y_point, c=rainb)
	plt.title('Tracking')
	ani2 = animation.FuncAnimation(fig, updatefig2, frames=129, fargs=(fig, im, centroid), interval=30)



	plt.show()
	y_ = []
	x_ = []
	plt.subplot(121)
	rainb = cm.rainbow(np.linspace(0, 1, 2))
	entire_list = [x for x in range(len(df.columns))]
	evensList = entire_list[0::2]
	oddList = entire_list[1::2]
	for k in range(2):
		y = df.iloc[k, evensList]
		x = df.iloc[k, oddList]
		plot_trajectory(x,y,rainb[k])
	plt.show()

def plot_trajectory(x,y,c):
	plt.plot(x, y, color=c, linewidth=2)

def convert_rgb(im_):
	im_ = 0.3*im_[:,:,0]+0.59*im_[:,:,1]+0.11*im_[:,:,2]
	return im_


def updatefig(i, fig, scat):
	x_point = df[df.iloc[:, 2 * i] > -1000].iloc[:, 2 * i]
	i_list.append(i)
	x_list.append(len(x_point))
	scat.set_offsets(np.column_stack(([i_list, x_list])))
	return scat,

def updatefig2(i, fig, im, centroid):
	y_point = df[df.iloc[:, 2*i] > -1000].iloc[:, 2*i]
	x_point = df[df.iloc[:, 2*i+1] > -1000].iloc[:, 2*i+1]
	im.set_array(image[i])
	centroid.set_offsets(np.column_stack(([x_point, y_point])))
	return im, centroid,

main()