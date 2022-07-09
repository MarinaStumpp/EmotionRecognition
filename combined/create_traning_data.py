# imports
import os
import shutil

dir = 'org-data'

# iterate all folders
subfolders = [f.path for f in os.scandir(dir) if f.is_dir()]

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
                    img_dir = 'training_data/neutral'
                    shutil.copy(f, os.path.join(img_dir))
                elif name_split[2] == '02':
                    img_dir = 'training_data/calm'
                    shutil.copy(f, os.path.join(img_dir))
                elif name_split[2] == '03':
                    img_dir = 'training_data/happy'
                    shutil.copy(f, os.path.join(img_dir))
                elif name_split[2] == '04':
                    img_dir = 'training_data/sad'
                    shutil.copy(f, os.path.join(img_dir))
                elif name_split[2] == '05':
                    img_dir = 'training_data/angry'
                    shutil.copy(f, os.path.join(img_dir))
                elif name_split[2] == '06':
                    img_dir = 'training_data/fearful'
                    shutil.copy(f, os.path.join(img_dir))
                elif name_split[2] == '07':
                    img_dir = 'training_data/disgust'
                    shutil.copy(f, os.path.join(img_dir))
                elif name_split[2] == '08':
                    img_dir = 'training_data/surprised'
                    shutil.copy(f, os.path.join(img_dir))
