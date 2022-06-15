# imports
import cv2
import os
import time

dir = 'org-data'

# iterate all folders
subfolders = [ f.path for f in os.scandir(dir) if f.is_dir() ]

for folder in subfolders:

	# iterate all files in folder
	for filename in os.listdir(folder):
		f = os.path.join(folder, filename)
		
		# checking if it is a file
		if os.path.isfile(f):
			
			# if 01 as first digit
			name_split = f.split("\\")
			name_split = name_split[2].split('-')
			
			if name_split[0] == '01':
				print(f)
				
				# set the folder
				if name_split[2] == '01':
					img_dir = 'C:\\Users\Philipp\\Desktop\\ER-Projekt\\data\\neutral'
				elif name_split[2] == '02':
					img_dir = 'C:\\Users\Philipp\\Desktop\\ER-Projekt\\data\\calm'
				elif name_split[2] == '03':
					img_dir = 'C:\\Users\Philipp\\Desktop\\ER-Projekt\\data\\happy'
				elif name_split[2] == '04':
					img_dir = 'C:\\Users\Philipp\\Desktop\\ER-Projekt\\data\\sad'
				elif name_split[2] == '05':
					img_dir = 'C:\\Users\Philipp\\Desktop\\ER-Projekt\\data\\angry'
				elif name_split[2] == '06':
					img_dir = 'C:\\Users\Philipp\\Desktop\\ER-Projekt\\data\\fearful'
				elif name_split[2] == '07':
					img_dir = 'C:\\Users\Philipp\\Desktop\\ER-Projekt\\data\\disgust'
				elif name_split[2] == '08':
					img_dir = 'C:\\Users\Philipp\\Desktop\\ER-Projekt\\data\\surprised'
				
				'''
				f = open(complete_dir, "w")
				f.write("This text is written in python")
				f.close
				'''
				
				# for each frame in in mp4 file => move image to corresponding folder
				count = 0
				vidcap = cv2.VideoCapture(f)
				success, image = vidcap.read()
				while success:
					if count%3 == 0:
						cv2.imwrite(img_dir + '\\' + f'{name_split[2]}-{time.time()}.jpg', image)
						print(count)
					success,image = vidcap.read()
					print('Read a new frame: ', success)
					count+=1
					