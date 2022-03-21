import os
import json
import argparse
import collections
import functools
import operator
from PIL import Image
from scipy.stats import norm
import glob
import shutil
import cv2

from collections import defaultdict

from vis import *
import numpy as np
import requests

import glob
from shutil import copyfile


set_of_jsons = {}
tr = 0

def parse_args():
    parser = argparse.ArgumentParser(description='Generate the clip for testing VOS')
    parser.add_argument('--a', type=str, default='annotations.json', help='annotations information')
    parser.add_argument('--v', type=str, default='check_nouns/maps.pkl',
                        help='video path')
    parser.add_argument('--m', type=str, default='map.json', help='the map between frame id and video id')
    parser.add_argument('--img', type=str, default='./selected_frames_for_Toronto/ori', help='frame root path')
    parser.add_argument('--vis', type=bool, default=True, help='vis masks')
    parser.add_argument('--v55', type=str,
                        default='/run/user/1000/gvfs/smb-share:server=rdsfcifs.acrc.bris.ac.uk,share=epickitchens/Dataset',
                        help='videos path of EPIC-KITCHENS 55.')
    parser.add_argument('--v100', type=str,
                        default='//run/user/1000/gvfs/smb-share:server=rdsfcifs.acrc.bris.ac.uk,share=epickitchens/Dataset_2020',
                        help='videos path of EPIC-KITCHENS 100.')
    parser.add_argument('--d', type=int, default=60, help='duration of generated clip (s)')
    parser.add_argument('--s', type=str, default='./clips', help='save path of generated clip')
    return parser.parse_args()


def find_video_path(args, vid):
    pid = vid.split('_')[0]
    if len(vid.split('_')[1]) == 3:
        path = os.path.join(args.v100, pid, "videos", vid+".MP4")
    elif len(vid.split('_')[1]) == 2:
        path = os.path.join(args.v55, pid, "videos", vid+".MP4")
    else:
        raise
    return path

def get_frame_info(annotation, map_info):
    if not annotation['annotation']['annotationGroups'][0]['annotationEntities'][0]['annotationBlocks'][0]['annotations']:
        return None, None
    imgId = annotation['annotation']['annotationGroups'][0]['annotationEntities'][0]['annotationBlocks'][0]['annotations'][0]['image']
    for i in range(len(annotation['documents'])):
        if annotation['documents'][i]['id'] == imgId:
            fid = annotation['documents'][i]['name']
            anno = annotation['documents'][i]['directory']
            break
    idx = int(fid[6:16]) - 1
    fid = map_info[anno][idx]
    vid = '_'.join(fid.split('_')[:2])
    fid = '_'.join(fid.split('_')[2:])
    return vid, fid

def get_masks_info(annotation):
    masks = defaultdict(list)
    for entity in annotation['annotation']['annotationGroups'][1]['annotationEntities']:
        masks['entity'].append(entity['name'])
        masks['masks'].append(entity["annotationBlocks"][0]["annotations"])
    return masks

def generate_clip(args, vid, fid):
    #/Users/ahmaddarkhalil/Downloads/P30_107.mp4
    vpath = find_video_path(args, vid)
    start_idx = int(fid[6:16]) -1
    fps = 60 if len(vid.split('_')[-1]) == 2 else 50
    call = "ffmpeg -i {} - vf \'select=between(n\,{}\,{})\' - y - acodec copy. / {}.mp4".format(
        vpath,
        start_idx,
        start_idx+fps*args.d,
        os.path.join(args.s+"/"+vid+"_"+fid[6:16])
    )
    os.system(call)

def generate_images(video_path,dest_folder):
    x = "ffmpeg -i "+video_path+" -qscale:v 1 -qmin 1 -r 50 "+dest_folder+"frame_%10d.jpg"
    os.system(x)

def generate_images_n(video_path,dest_folder,start_frame_id,number_of_frames):
    start_idx = int(start_frame_id[6:16]) - 1
    x = "ffmpeg -i "+video_path+" -vf \'select=between(n\,"+str(start_idx)+"\,"+str(start_idx+number_of_frames)+")\' -qscale:v 1 -vsync 0 -qmin 1 -r 50 "+dest_folder+"frame_%10d.jpg"
    print(x)
    os.system(x)

def generate_images_n_V2(video_path,dest_folder,start_frame_id,number_of_frames):
    #this function generates N number of images (frames) of a video. start_frame_id: suppose video starts with frame#1 as 'generate_images' works
    start_idx = int(start_frame_id.split("_")[-1][:-4]) - 1 # minus 1 since the following function suppose first frame is 0 where as when 'generate_images' started with frame 1
    prefix = ''
    if len(start_frame_id.split('_')) == 5: #if the name contains the video and the video id prefix (ee.g. 00094)
        prefix = '_'.join(start_frame_id.split('_')[0:3]) + '_'
    else:
        prefix = ''

    vid = video_path.split("/")[-1].split('.')[0]
    frame_rate = 60 if len(vid.split('_')[-1]) == 2 else 50
    #-vf scale=854:480
    start_idx = start_idx/frame_rate
    x = "/mnt/storage/home/ru20956/scratch/ffmpeg-4.4-amd64-static/ffmpeg -loglevel error -threads 16 -ss "+str(start_idx)+" -i "+video_path+" -qscale:v 2 -vf scale=854:480 -frames:v  "+str(number_of_frames)+" "+dest_folder+"/"+prefix+"frame__%10d.jpg"
    print(x)
    os.system(x)

def generate_images_n_V2_index1(video_path,dest_folder,start_frame_id,number_of_frames):
    #this function generates N number of images (frames) of a video. start_frame_id: suppose video starts with frame#1 as 'generate_images' works
    start_idx = int(start_frame_id.split("_")[-1][:-4]) - 1 # minus 1 since the following function suppose first frame is 0 where as when 'generate_images' started with frame 1
    print(start_idx)
    start_idx = int(start_frame_id.split("_")[-1][:-4]) - 1 # minus 1 since the following function suppose first frame is 0 where as when 'generate_images' started with frame 1
    prefix = ''
    if len(start_frame_id.split('_')) == 5: #if the name contains the video and the video id prefix (ee.g. 00094)
        prefix = '_'.join(start_frame_id.split('_')[0:3]) + '_'
    else:
        prefix = ''
    vid = video_path.split("/")[-1].split('.')[0]
    frame_rate = 60 if len(vid.split('_')[-1]) == 2 else 50

    start_idx = start_idx/frame_rate
    x = "/mnt/storage/home/ru20956/scratch/ffmpeg-4.4-amd64-static/ffmpeg -loglevel error -ss "+str(start_idx)+" -i "+video_path+" -qscale:v 2 -vf scale=854:480 -frames:v  "+str(number_of_frames)+" "+dest_folder+'/'+prefix+"frame__%10d.jpg"
    print(x)
    os.system(x)

def generate_images_n_V2_index2(video_path,seq_name,dest_folder,start_frame_id,number_of_frames):
    #this function generates N number of images (frames) of a video. start_frame_id: suppose video starts with frame#1 as 'generate_images' works
    start_idx = int(start_frame_id.split("_")[-1][:-4]) - 1 # minus 1 since the following function suppose first frame is 0 where as when 'generate_images' started with frame 1
    print(start_idx)
    start_idx = start_idx/50.0
    x = "ffmpeg -loglevel error -ss "+str(start_idx)+" -i "+video_path+" -qscale:v 1 -qmin 1 -frames:v  "+str(number_of_frames)+" "+dest_folder+"/"+seq_name+".mp4"
    print(x)
    os.system(x)

def generate_images_n_V2_with_labels(video_path,seq_name,dest_folder,start_frame_id,number_of_frames,obj_list,action_list):
    start_idx = int(start_frame_id.split("_")[-1][:-4]) - 1 # minus 1 since the following function suppose first frame is 0 where as when 'generate_images' started with frame 1
    print(start_idx)
    start_idx = start_idx/50.0
    x = 'ffmpeg -threads 4 -loglevel error -ss ' + str(start_idx) + ' -i ' + video_path + ' -qscale:v 4 -qmin 4 -vf drawtext="fontfile=FreeSerif.otf: \
       text=' + "\n".join(obj_list) + ': fontcolor=white: fontsize=54: box=1: boxcolor=black@0.8: \
       boxborderw=10: x=(50): y=(100), drawtext=fontfile=FreeSerif.otf: \
       text=' + "\n".join(action_list) + ': fontcolor=white: fontsize=54: box=1: boxcolor=black@0.8: \
       boxborderw=10: x=(w-text_w-50): y=(100)" -frames:v  ' + str(
        number_of_frames) + ' ' + dest_folder + '/' + seq_name + '.mp4'
    os.system(x)
    #os.system('ffmpeg -i '+dest_folder+'/'+seq_name+'.mp4'+'.mp4 -vf drawtext="fontfile=FreeSerif.otf: \
    #text=' + "\n".join(obj_list) + ': fontcolor=white: fontsize=54: box=1: boxcolor=black@0.7: \
    #boxborderw=5: x=(w-text_w)/10: y=(h-text_h)/10" -codec:a copy output.mp4')
    print(x)

def generate_images_n_V3_index1(video_path,dest_folder,start_frame_id,number_of_frames):
    #this function generates N number of images (frames) of a video. start_frame_id: suppose video starts with frame#1 as 'generate_images' works
    start_idx = start_frame_id - 1 # minus 1 since the following function suppose first frame is 0 where as when 'generate_images' started with frame 1
    start_idx = start_idx/50.0
    x = "ffmpeg -loglevel error -ss "+str(start_idx)+" -i "+video_path+" -qscale:v 1 -qmin 1 -frames:v  "+str(number_of_frames)+" "+dest_folder+"/%05d.jpg"
    print(x)
    os.system(x)

def generate_epic_train_dataset_info(filename, vid, selected_frame, output_directory):
    selected_frame_int = int(selected_frame[6:16]) #get the selected frame as int
    import pandas as pd
    df = pd.read_csv(filename)
    df = df[["video_id", "start_frame", "stop_frame"]].dropna() #select the important colunms for this task
    df = df[df.video_id == vid] #filter the dataset to just include the video we're targeting
    df = df[(df.start_frame <= selected_frame_int) & (df.stop_frame >= selected_frame_int)]
#    if (df.__len__() != 1):
#        print("The frame does not found or occures more than one time in the csv file!!!")
#        exit()

    start_frame = selected_frame_int #let the start frame to be the selected frame from the segmentation
    stop_frame = df.iloc[0].stop_frame #let the stop frame to be the final frame in the action
    number_of_frames = stop_frame - start_frame + 1 #both first and last frames are included

    generate_images_n_V2('/Users/ahmaddarkhalil/Downloads/P30_107.mp4', output_directory,selected_frame,number_of_frames)


def annotations_to_datatset(filename, path_to_store):
    import os
    import json
    # Opening JSON file
    f = open(filename)
    # returns JSON object as a dictionary
    data = json.load(f)

    # Iterating through the json list
    for datapoint in data:
        directory_name = datapoint['documents'][0]['directory'] #directory name of the action
        if os.path.exists(os.path.join(path_to_store, directory_name)) == False: # if the sub-folders does not exists, then create them
            os.mkdir(os.path.join(path_to_store, directory_name)) #make directory to store the action images
        vid = directory_name.split("-")[0]  # get the vid like P30_107 out of the first image in the datapoint with P30_107-take_plate_002 format
        image_id = datapoint['annotation']['annotationGroups'][0]['annotationEntities'][0]['annotationBlocks'][0]['annotations'][0]['image'] #get the ID of the selected image
        frame_number = '' #initailize frame number
        for document in datapoint['documents']:
            if (document['id'] == image_id): #check the IDs
                frame_number = document['name'] #get the frame which would be like frame_0000066642.jpg
                print(frame_number)
                break
        generate_epic_train_dataset_info('EPIC_100_train.csv', vid, frame_number, os.path.join(path_to_store, directory_name))


def generate_N_sequence_of_actions(video_path,json_file,id,vid, csv_info_file,out_fps,number_of_seq,output_dir,actions_per_sequence=3):
    import pandas as pd
    filter_hands_objects(json_file) # update the file to include both left and right hands
    df = pd.read_csv(csv_info_file)
    frame_rate = 60 if len(vid.split('_')[-1]) == 2 else 50
    print("Video name: ", vid, " with ", frame_rate, "fps")
    df = df[df.video_id == vid]  # filter the dataset to just include the video we're targeting
    df = df[["video_id", "start_frame", "stop_frame"]].dropna() #select the important colunms for this task
    df = df.sort_values(by=['stop_frame'])
    total_objects = df.start_frame.count()
    i = 0
    sequence_limits = []
    count_zeros=0
    sum_non_z=0
    row=[]
    sum_duration = 0
    total_files_per_folder=0
    folder_index = 1
    sum_non_z_c=0
    output_dir_part = vid+'_Part_' + str(folder_index).zfill(3)
    os.mkdir(os.path.join(output_dir,output_dir_part))
    while not df.empty:
        seq = df.head(actions_per_sequence)
        #print(df.head(9))
        min_value = seq.start_frame.min()
        max_value = seq.stop_frame.max()
        row= [min_value,max_value]
        sequence_limits.append([min_value,max_value])
        #print(seq, "\nMIN, MAX : ",min_value, max_value)
        #print("S" + str(i) + "=> " + str(min_value) + ": " + str(max_value))
        df = df[(df.start_frame > max_value)]


        if ((row[1]-row[0]) <= 0):
            print("ERROR!!, start frame of seq = ", row[0], "and end =", row[1])
            exit(1)

        generate_images_n_V2('/Users/ahmaddarkhalil/Downloads/P30_107.mp4', './P30_107_n/', 'frame_0000000001.jpg', 3)

        start=row[0]
        stop=row[1]
        objects_set = set()
        # Opening JSON file
        f = open(json_file)
        # returns JSON object as a dictionary
        data = json.load(f)
        # Iterating through the json list
        for folder_name in data.keys():
            frame_number = int(folder_name.split('/')[-1].split('_')[-1])
            if(frame_number >= start and frame_number <= stop):
                objects = data[folder_name]
                #print("ADD ",objects_set)
                for object in objects["labels"]:
                    objects_set.add(object)

        print("Final set ", objects_set.__len__())
        if(objects_set.__len__() == 0):
            count_zeros = count_zeros + 1
        else:
            sum_non_z = sum_non_z + objects_set.__len__()

            sum_duration =sum_duration + (row[1]-row[0] + 1)/frame_rate # there we calculate the duration of an action, +1 since all inclusive,
            print("Time of the sequence: ", (row[1]-row[0] + 1)/frame_rate)
            print("S" + str(i+1) + "=> " + str(row[0]) + ": " + str(row[1]))
            output_name = id+"_"+vid+"_seq_"+str(i+1).zfill(5)
            print(output_name)
            # Path
            path = os.path.join(output_dir+output_dir_part,output_name)
            if not os.path.exists(path):
                os.mkdir(path,mode=0o777)
            frame_name = id+"_"+vid+'frame_'+str(start).zfill(10)+'.jpg'
            print(frame_name)
            generate_images_n_V2_index1(video_path, path, frame_name, stop-start+1)
            num_of_files = rename_and_resample_files(path,start,stop,objects_set)
            sum_non_z_c = sum_non_z_c + num_of_files
            total_files_per_folder = total_files_per_folder + num_of_files


            i = i + 1
            if(i == number_of_seq):
                break
            if (total_files_per_folder > 180):
                folder_index = folder_index + 1
                output_dir_part = vid+'_Part_' + str(folder_index).zfill(3)
                output_json_part = vid+'_Part_' + str(folder_index-1).zfill(3)
                global set_of_jsons
                file = open(output_json_part+".json", "w")
                a = json.dumps(set_of_jsons)
                file.write(str(a))  # delete the first and last char (array []) as required in the json file
                file.close()
                set_of_jsons = {}
                os.mkdir(os.path.join(output_dir, output_dir_part))

                print("Number of files in this folder: ",total_files_per_folder)
                total_files_per_folder = 0

    output_json_part = vid+'_Part_' + str(folder_index).zfill(3)
    file = open(output_json_part + ".json", "w")
    a = json.dumps(set_of_jsons)
    file.write(str(a))
    file.close()


    average_seq_duration = sum_duration/len(sequence_limits)
    sum_of_all_seq_frames = round(sum_duration*frame_rate,0)

    print("Average duration per sequence: ",average_seq_duration, "seconds")
    print("Number of all annotations: ",sum_non_z_c)
    print("Number of sequences: ", i)
    print("Json file length = ", len(data)/4)
    print("Number of actions in the csv file: ",total_objects)
    print("Number of missing actions within sequence in json file: ", count_zeros)
    print("Number of  objects: ", sum_non_z)


def rename_and_resample_files(path,start,stop,objects_set):
    num_of_files=0
    for i in range(1,stop-start+2):
        frame_name = 'frame_' + str(i).zfill(5) + '.jpg'
        if (i%25 == 1 or i == stop-start+1): #sampling with 2fps, the second condition to include the last frame always
            new_frame_name = 'frame_' + str(start+i-1).zfill(10) +'/frame_' + str(start+i-1).zfill(10) + '.jpg'
            os.mkdir(os.path.join(path,'frame_' + str(start+i-1).zfill(10))) # to create folder for each image
            os.rename(os.path.join(path,frame_name), os.path.join(path,new_frame_name))
            print(os.path.join(path,new_frame_name))
            print(objects_set)
            num_of_files = num_of_files + 1
            m = {"labels": list(objects_set)}
            set_of_jsons[os.path.join(path,new_frame_name)] = m

            #set_of_jsons = json.loads(set_of_jsons)
        else:
            os.remove(os.path.join(path,frame_name))
    return num_of_files


def rename_and_resample_masks(path,extention):

    index = 0
    store_index=0
    for subdir, dirs, files in sorted(os.walk(path)):
        index = 0
        store_index = 0
        if (subdir != path and "_seq_" in subdir): #make sure that the subdir is belong to a sequance
            for infile in sorted(glob.glob(subdir+'/*.'+extention)):
                if (index % 25 == 0 or index == len(files) -1):
                    path_2fps = subdir.replace("seq", "seq_50_to_2fps")
                    os.makedirs(path_2fps, exist_ok=True, mode=0o777)
                    store_path = os.path.join(path_2fps,'{0}'.format(str(store_index).zfill(5))+"."+extention)
                    copyfile(infile, store_path)
                    print(infile,"=>",store_path)
                    store_index = store_index + 1
                index = index + 1


def generate_sequence_of_actions(video_path,vid, csv_info_file,actions_per_sequence=3):
    import pandas as pd
    df = pd.read_csv(csv_info_file)
    frame_rate = 60 if len(vid.split('_')[-1]) == 2 else 50
    print("Video name: ", vid, " with ", frame_rate, "fps")
    df = df[df.video_id == vid]  # filter the dataset to just include the video we're targeting
    df = df[["video_id", "start_frame", "stop_frame"]].dropna() #select the important colunms for this task
    df = df.sort_values(by=['stop_frame'])

    actions_count = df.stop_frame.count()
    i = 0
    sequence_limits = []
    while not df.empty:
        seq = df.head(actions_per_sequence)
        #print(df.head(9))
        min_value = seq.start_frame.min()
        max_value = seq.stop_frame.max()
        sequence_limits.append([min_value,max_value])
        #print(seq, "\nMIN, MAX : ",min_value, max_value)
        #print("S" + str(i) + "=> " + str(min_value) + ": " + str(max_value))
        df = df[(df.start_frame >= max_value)]
        #print(df.describe())
        i = i + 1
    sum_duration = 0
    for index, row in enumerate(sequence_limits):
        #print("S" + str(index) + "=> " + str(row[0]) + ": " + str(row[1]))
        if ((row[1]-row[0]) <= 0):
            print("ERROR!!, start frame of seq = ", row[0], "and end =", row[1])
            exit(1)
        sum_duration =sum_duration + (row[1]-row[0] + 1)/frame_rate # there we calculate the duration of an action, +1 since all inclusive,
        #print((row[1]-row[0] + 1)/frame_rate)

    average_seq_duration = sum_duration/len(sequence_limits)
    sum_of_all_seq_frames = round(sum_duration*frame_rate,0)
    percentage_of_deleted_actions = (1 - (len(sequence_limits))/(actions_count/actions_per_sequence))*100

    print("Average duration per sequence: ",average_seq_duration, "seconds")
    print("Sum of all frames of all sequences: ",sum_of_all_seq_frames)

    if (percentage_of_deleted_actions < 0):
        #print("ERROR!!, #of seq = ", len(sequence_limits), "and action count= ", actions_count, "actions_per_sequence = ",actions_per_sequence)
        #exit(1)
        percentage_of_deleted_actions = 0 # in this case all actions are included e.g len(sequence_limits) = 2,actions_count=5,actions_per_sequence=3
    print("Percentage of deleted actions: ", percentage_of_deleted_actions, "%")
    print("Number of sequences: ", i)
    return average_seq_duration,sum_of_all_seq_frames,percentage_of_deleted_actions,i

def generate_sequence_of_actions_all_dataset(video_path,csv_info_file,actions_per_sequence=3):
    import pandas as pd
    import numpy as np
    df_all = pd.read_csv(csv_info_file)
    stats = []
    for subset in df_all.video_id.unique():
        average_seq_duration,sum_of_all_seq_frames,percentage_of_deleted_actions,i = generate_sequence_of_actions(video_path,subset,csv_info_file,actions_per_sequence)
        stats.append([average_seq_duration,sum_of_all_seq_frames,percentage_of_deleted_actions,i])

    stats_arr = np.array(stats)
    average_seq_duration = sum(stats_arr[:,0])/len(stats)
    sum_of_all_seq_frames = sum(stats_arr[:,1])
    average_percentage_of_deleted_actions = sum(stats_arr[:,2])/len(stats)
    total_number_of_seq = sum(stats_arr[:, 3])
    #np.save('N_3.npy',stats_arr)
    print("Global Average duration per sequence: ",average_seq_duration, "seconds")
    print("Min:", min (stats_arr[:,0]), "\nMAX:",max (stats_arr[:,0]))
    print("Global Sum of all frames of all sequences: ",sum_of_all_seq_frames)
    print("Min:", min(stats_arr[:, 1]), "\nMAX:", max(stats_arr[:, 1]))
    print("Global Percentage of deleted actions: ", average_percentage_of_deleted_actions,"%")
    print("Min:", min(stats_arr[:, 2]), "\nMAX:", max(stats_arr[:, 2]))
    print("Global total number of sequences : ", sum(stats_arr[:, 3]), ", average per video", sum(stats_arr[:, 3]) / len(stats))


    #visualize_stats(stats_arr)

def visualize_stats(stats_arr):
    import numpy as np
    import random
    from matplotlib import pyplot as plt
    data = stats_arr[:,0]
    plt.hist(data, bins=40, alpha=0.5)
    plt.title('Average sequence durations')
    plt.ylabel('Count')
    plt.xlabel('Duration')
    plt.show()
    data = stats_arr[:, 1]
    plt.hist(data, bins=40, alpha=0.5)
    plt.title('Number of frames per video (all sequences)')
    plt.ylabel('Count')
    plt.xlabel('Number of frames')
    plt.show()
    data = stats_arr[:, 2]
    plt.hist(data, bins=40, alpha=0.5)
    plt.title('Percentage of deleted actions')
    plt.ylabel('Count')
    plt.xlabel('Percentage')
    plt.show()


def filter_hands_objects (filename):
    import json
    f = open(filename)
    # returns JSON object as a dictionary
    data = json.load(f)
    # Iterating through the json list
    for key in data.keys():
        object = data[key]
        # print("ADD ",objects_set)
        if object["labels"].count("hand") > 0:
            object["labels"].remove("hand")

        if object["labels"].count("left hand") == 0:
            object["labels"].append("left hand")

        if object["labels"].count("right hand") == 0:
            object["labels"].append("right hand")
    f.close()
    out_file = open(filename, "w")
    json.dump(data, out_file)
    out_file.close()
    print("left and right hand inserted to the json file, Done!!")


def json_to_masks(filename,output_directory,object_keys=None):
    import os
    os.makedirs(output_directory, exist_ok=True)
    import json
    f = open(filename)
    # returns JSON object as a dictionary
    data = json.load(f)
    #sort based on the folder name (to guarantee to start from its first frame of each sequence)
    data = sorted(data, key=lambda k: k['documents'][0]['directory'])
    # Iterating through the json list
    full_path=""
    for datapoint in data:
        image_name = datapoint["documents"][0]["name"]
        image_path = datapoint["documents"][0]["directory"]
        masks_info = datapoint["annotation"]["annotationGroups"][0]["annotationEntities"]
        full_path =output_directory+image_path.split('/')[0]+'/' #until the end of sequence name
        #print(full_path)
        os.makedirs(full_path,exist_ok= True)
        #generate_masks_stage3(image_name, image_path, masks_info, full_path) #this is for saving the same name (delete the if statemnt as well)
        #generate_masks_per_seq(image_name, image_path, masks_info, full_path)
        generate_masks(image_name, image_path, masks_info, full_path,object_keys) #this is for unique id for each object throughout the video

def folder_of_jsons_to_masks(input_directory,output_directory):
    import glob
    import csv
    import pandas as pd
    import os
    objects_keys = {}
    for json_file in sorted(glob.glob(input_directory + '/*.json')):
        objects_keys = {}
        objects = do_stats_stage2_jsons_single_file(json_file)
        #print('objects: ',objects)
        i = 1
        for key,_ in objects:
            objects_keys[key] = i
            i=i+1
        max_count = max(objects_keys.values())
        print(f'unique object count of {json_file.split("/")[-1]} is {max_count}')
        if max_count > 100:
            print('Number of unique objects is more than 100!!!!!')
            #exit(0)
        json_to_masks(json_file,output_directory,objects_keys)


        data = pd.DataFrame(objects_keys.items(), columns=['Object_name', 'unique_index'])
        data['video_id'] = int(json_file.split('/')[-1].split('.')[0].split('_')[0])
        if not os.path.isfile('data_mapping.csv'):
            data.to_csv('data_mapping.csv', index=False,header=['Object_name', 'unique_index','video_id'])
        else:
            data.to_csv('data_mapping.csv',mode='a', header=False,index=False)



def generate_videos_from_all_masks(input_directory,output_directory,frame_rate):
    import glob
    import os
    os.makedirs(output_directory,exist_ok=True)
    prev_video = ""
    for video in sorted(glob.glob(input_directory + '/*')):
        if ("_".join(video.split("/")[-1].split("_")[:-2]) != prev_video):
            seq_to_video_modified(video.split("/")[0]+"/"+"_".join(video.split("/")[-1].split("_")[:-2]),output_directory,frame_rate) #video like "'segmentations/0002_P06_101'"
            prev_video = "_".join(video.split("/")[-1].split("_")[:-2])

def generate_videos_from_all_masks_vis(input_directory,frame_rate):
    import glob
    import os
    for video in sorted(glob.glob(input_directory + '/*')):
            seq_to_video_modified_vis(video,frame_rate) #video like "'segmentations/0002_P06_101'"

def copy_jpg_images(input_directory):
    import glob
    import os
    import shutil
    from PIL import Image
    for video in sorted(glob.glob(input_directory + '/*')):
        shutil.rmtree(video+"/"+"original_frames")
        os.makedirs(video+"/"+"original_frames",exist_ok=True)
        for images in sorted(glob.glob(video + '/mask_frames/*.png')):
            folder="_".join(images.split('/')[1].split('_')[1:3])
            image = images.split('/')[-1].split('_')[-1][:-4]
            print(glob.glob(folder + '/*/*/*/*'+image+"*")[0].replace("jpg","png"))
            im1 = Image.open(glob.glob(folder + '/*/*/*/*'+image+"*")[0])
            im1.save(video+"/"+"original_frames/"+image+".png")
            #shutil.copy(glob.glob(folder + '/*/*/*/*'+image+"*")[0],video+"/"+"original_frames/"+image+".png")



def json_to_masks_vos(filename,output_directory):
    import json
    import shutil
    f = open(filename)
    # returns JSON object as a dictionary
    data = json.load(f)
    #sort based on the folder name (to guarantee to start from its first frame of each sequence)
    data = sorted(data, key=lambda k: k['documents'][0]['directory'])
    #this list to check the start of the sequence, max number of seq per Pxx_xx file is 1000
    directories= ""
    objects_in_first_frame = []
    # Iterating through the json list
    for datapoint in data:
        image_name = datapoint["documents"][0]["name"]
        image_path = datapoint["documents"][0]["directory"]
        masks_info = datapoint["annotation"]["annotationGroups"][0]["annotationEntities"]
        full_path =output_directory+ "/masks/"+"/".join(image_path.split('/')[2:-2])+"/" #until the end of sequence name

        #copy the image files
        full_path_jpg = output_directory + "/images/"+"/".join(image_path.split('/')[2:-2])  # until the end of sequence name
        print(full_path_jpg)
        os.makedirs(full_path_jpg, exist_ok=True,mode=0o777)
        shutil.copy(image_path, full_path_jpg)

        is_before = (directories == full_path) # to check if the current image belongs to the same seq or different one
        if is_before == False:
            directories = full_path
            os.makedirs(full_path,exist_ok= True,mode=0o777)
            objects_in_first_frame = generate_masks_vos_edited(image_name, image_path, masks_info, full_path,[])
        else:
            generate_masks_vos_edited(image_name, image_path, masks_info, full_path,objects_in_first_frame)

def resize_images(folder):
    import os
    import glob
    import cv2
    import numpy as np
    index=0
    for infile in sorted(glob.glob(folder+'/*.jpg')):
        img = cv2.imread(infile)
        resized = cv2.resize(img, (854, 480), interpolation=cv2.INTER_NEAREST)
        path480 = folder+'_480'
        os.makedirs(path480, exist_ok=True, mode=0o777)
        cv2.imwrite(os.path.join(path480,'{0}'.format(str(index).zfill(5))+'.jpg'),resized)
        print ("Current File Being Processed is: " + infile)
        index=index+1

def resize_all_images(folder):
    import os
    import glob
    import cv2
    import numpy as np
    index = 0
    for subdir, dirs, files in sorted(os.walk(folder)):
        index = 0
        if (subdir != folder):
            path480 = subdir.replace("images", "images_480")
            print(path480)
            for infile in sorted(glob.glob(subdir+'/*.jpg')):
                img = cv2.imread(infile)
                resized = cv2.resize(img, (854, 480), interpolation=cv2.INTER_LINEAR)
                os.makedirs(path480, exist_ok=True, mode=0o777)
                cv2.imwrite(os.path.join(path480,infile.split("/")[-1]),resized)
                print ("Current File Being Processed is: " + infile)
                index=index+1

def resize_all_masks(folder):
    import os
    import glob
    import cv2
    import numpy as np
    from PIL import Image
    from numpy import asarray
    index = 0
    for subdir, dirs, files in sorted(os.walk(folder)):
        index = 0
        if (subdir != folder):
            path480 = subdir.replace("/masks/", "/masks_480/")
            print(path480)
            for infile in sorted(glob.glob(subdir+'/*.png')):
                mask = Image.open(infile)
                # convert image to numpy array
                data = asarray(mask)
                small_lable = cv2.resize(data, (854,
                                                480),
                                         interpolation=cv2.INTER_NEAREST)
                small_lable = (np.array(small_lable)).astype('uint8')
                print(small_lable.shape, np.unique(small_lable))
                os.makedirs(path480, exist_ok=True, mode=0o777)
                # cv2.imwrite(os.path.join(path480,'{0}'.format(str(index).zfill(5))+'.png'), small_lable)
                imwrite_indexed(os.path.join(path480,infile.split("/")[-1]), small_lable)
                print("Current File Being Processed is: " + infile)
                index = index + 1



def resize_masks(folder):
    import numpy as np
    import cv2
    from PIL import Image
    from numpy import asarray
    import glob
    index = 0
    for infile in sorted(glob.glob(folder + '/*.png')):
        mask = Image.open(infile)
        # convert image to numpy array
        data = asarray(mask)
        small_lable = cv2.resize(data, (854,
                                 480),
                                interpolation=cv2.INTER_NEAREST)
        small_lable = (np.array(small_lable)).astype('uint8')
        print(small_lable.shape, np.unique(small_lable))
        path480 = folder + '_480'
        os.makedirs(path480, exist_ok=True, mode=0o777)
        #cv2.imwrite(os.path.join(path480,'{0}'.format(str(index).zfill(5))+'.png'), small_lable)
        imwrite_indexed(os.path.join(path480,'{0}'.format(str(index).zfill(5))+'.png'),small_lable)
        print ("Current File Being Processed is: " + infile)
        index=index+1
def imwrite_indexed(filename, im):
    from PIL import Image
    davis_palette = np.repeat(np.expand_dims(np.arange(0, 256), 1), 3, 1).astype(np.uint8)
    davis_palette[:22, :] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                             [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                             [64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],
                             [64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],
                             [0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],
                             [0, 64, 128], [128, 64, 128]]

    color_palette = davis_palette
    assert len(im.shape) < 4 or im.shape[0] == 1  # requires batch size 1
    im = torch.from_numpy(im)
    im = Image.fromarray(im.detach().cpu().squeeze().numpy(), 'P')
    im.putpalette(color_palette.ravel())
    im.save(filename)

def generate_flat_data(folder):
    import glob
    import os
    import shutil
    os.makedirs(folder + '_flat', exist_ok=True, mode=0o777)

    for infile in sorted(glob.glob(folder + '/*/*/*/*.jpg')):
        full_path_jpg_folder = "/".join(infile.split('/')[:-3])
        path480 = os.path.join(folder + '_flat',full_path_jpg_folder)
        os.makedirs(path480, exist_ok=True, mode=0o777)
        print(full_path_jpg_folder)
        full_path_jpg = infile.replace('/','_')
        shutil.copy(infile, os.path.join(path480,full_path_jpg))
        print(os.path.join(path480,full_path_jpg))

def generate_flat_data_png(folder):
    import glob
    import os
    import shutil
    os.makedirs(folder + '_flat', exist_ok=True, mode=0o777)

    for infile in sorted(glob.glob(folder + '/*/*.png')):
        full_path_jpg_folder = "/".join(infile.split('/')[:-3])
        path480 = os.path.join(folder + '_flat',full_path_jpg_folder)
        os.makedirs(path480, exist_ok=True, mode=0o777)
        print(full_path_jpg_folder)
        full_path_jpg = infile.replace('/','_')
        shutil.copy(infile, os.path.join(path480,full_path_jpg))
        print(os.path.join(path480,full_path_jpg))
def generate_flat_data_part_by_part_5d(folder):
    import glob
    import os
    import shutil
    os.makedirs(folder + '_flat', exist_ok=True, mode=0o777)
    i=0
    full_path_jpg_folder=""
    for infile in sorted(glob.glob(folder + '/*/*/*/*.jpg')):
        if full_path_jpg_folder != "/".join(infile.split('/')[:-3]):
            i=0
        full_path_jpg_folder = "/".join(infile.split('/')[:-3])
        path480 = os.path.join(folder + '_flat',full_path_jpg_folder)
        os.makedirs(path480, exist_ok=True, mode=0o777)
        print(full_path_jpg_folder)
        full_path_jpg = '{0}'.format(str(i).zfill(5))+".jpg"
        shutil.copy(infile, os.path.join(path480,full_path_jpg))
        print(os.path.join(path480,full_path_jpg))
        i=i+1
def generate_flat_data_all_video(folder):
    import glob
    import os
    import shutil
    os.makedirs(folder + '_flat_all', exist_ok=True, mode=0o777)

    for infile in sorted(glob.glob(folder + '/*/*/*/*.jpg')):
        full_path_jpg_folder = "/".join(infile.split('/')[:-5])
        path480 = os.path.join(folder + '_flat_all', full_path_jpg_folder)
        os.makedirs(path480, exist_ok=True, mode=0o777)
        print(full_path_jpg_folder)
        full_path_jpg = infile.replace('/', '_').split("_")[-1]
        shutil.copy(infile, os.path.join(path480, full_path_jpg))
        print(os.path.join(path480, full_path_jpg))

def measure_blur_image(path): #function to calculate the blur in each one of the sub-folders of a directory, it also displays the blured images
    import cv2
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
    return blur_value

def measure_blur(folder, threshold=70): #function to calculate the blur in each one of the sub-folders of a directory, it also displays the blured images
    import cv2
    import glob

    for subdir, dirs, files in sorted(os.walk(folder)):
        blur=[]
        count_blur = 0
        if subdir != folder:
            print(subdir.split("/")[-1])
            for infile in sorted(glob.glob(subdir + '/*.jpg')):
                image = cv2.imread(infile)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
                if (blur_value < threshold):
                    #print(infile,blur_value)
                    count_blur = count_blur + 1
                blur.append(blur_value)
            print("Average blur is ="+str(int(sum(blur)/len(blur))) + ", percentage of blurred images = " + format(100*(count_blur)/len(blur), '.3f')+" %")
    print("high value => less blur")


def generate_50fps_from_2fps(video_path,input_dir,output_dir):
    import glob
    import os
    #files in this input_dir are like:P30_107/P30_Part_001/P30_107_seq_00001/frame_0000000001/frame_0000000001.jpg

    seq_name = ""
    seq_start_frame_str=""
    seq_end_frame_str = ""
    seq_start_frame=0
    seq_end_frame = 0
    for infile in sorted(glob.glob(input_dir + '*/*/*.jpg')):
        if seq_name != infile.split("/")[-3]: #check if the current seq is not the same with previous one (new seq)
           if seq_name != "" : # if it is not the before starting (seq = "" as inital value)
                #print(len(os.walk("/".join(infile.split("/")[:-2])).__next__()[1]))
                #os.makedirs(os.path.join(output_dir,seq_name+"_"+str(seq_start_frame)+"_"+str(seq_end_frame)), exist_ok=True) this for start and end frame
                os.makedirs(os.path.join(output_dir,seq_name), exist_ok=True)
                print("Seq: ",seq_name)
                print("Start: ",seq_start_frame_str)
                print("Start: ",seq_start_frame)
                print("END: ",seq_end_frame)
                print("-----------------------")
                generate_images_n_V2(video_path, os.path.join(output_dir,seq_name), seq_start_frame_str,
                                     seq_end_frame - seq_start_frame + 1)
                rename_images(os.path.join(output_dir, seq_name), seq_start_frame,
                              seq_end_frame - seq_start_frame + 1)

                #assign the new values
                seq_start_frame_str = infile.split("/")[-1]
                seq_start_frame = int(seq_start_frame_str[-14:-4])
           else: # if it is the first iternation, then assign the start frame
                seq_start_frame_str = infile.split("/")[-1]
                seq_start_frame = int(seq_start_frame_str[-14:-4])



        seq_name = infile.split("/")[-3]  # set the new seq
        seq_end_frame_str = infile.split("/")[-1] #store each frame as a final frame until the above if statement comes true
        seq_end_frame = int(seq_end_frame_str[-14:-4])


    #for final folder
    print("Seq: ", seq_name)
    print("Start: ", seq_start_frame_str)
    print("Start: ", seq_start_frame)
    print("END: ", seq_end_frame)
    print("-----------------------")
    os.makedirs(os.path.join(output_dir,seq_name), exist_ok=True)
    generate_images_n_V2(video_path, os.path.join(output_dir,seq_name), seq_start_frame_str, seq_end_frame - seq_start_frame + 1)
    rename_images(os.path.join(output_dir,seq_name),seq_start_frame_str, seq_start_frame, seq_end_frame - seq_start_frame + 1)

def rename_images(path, seq_start_frame_str,start, stop):
    print(path)
    start_frame_id = seq_start_frame_str.split('/')[-1]
    prefix = ''
    if len(start_frame_id.split('_')) == 5: #if the name contains the video and the video id prefix (ee.g. 00094)
        prefix = '_'.join(start_frame_id.split('_')[0:3]) + '_'
    else:
        prefix = ''
    for i in range(1,stop+1):
        try:
            from_name = os.path.join(path,'{}frame__{}.jpg'.format(prefix,str(i).zfill(10)))
            to_name = os.path.join(path,'{}frame_{}.jpg'.format(prefix,str(start+i-1).zfill(10)))
            #print('From: ', from_name)
            #print('To: ', to_name)
            os.rename(from_name,to_name)
        except:
            break;
def combine_jsons_torento(folder):
    import json
    import glob
    all_annotations=[]
    for infile in sorted(glob.glob(folder + '/*.json')):
        f = open(infile, 'r')
        annotation_info = json.load(f)
        all_annotations.extend(annotation_info)

    print("Number of annotations: ",len(all_annotations))
    file  = open(folder+'/combined_annotations.json','w')
    all = json.dumps(all_annotations)
    file.write(str(all))
    file.close()

def display_seq_names():
    for subdir, dirs, files in sorted(os.walk("masks_vos_2/images")):
        if subdir != "masks_vos_2/images":
            print(subdir.replace("masks_vos_2/images/", ""))

def seq_to_video(filename,out_file,frame_rate):
    out_file = out_file+"/"+filename.split('/')[-1]
    os.system("ffmpeg -r "+str(frame_rate)+" -start_number 0 -i "+filename+"/*%010d.png -vcodec mpeg4 -qscale:v 1 -qmin 1 -y "+out_file+".mp4")
    #x = "ffmpeg -ss "+str(start_idx)+" -i "+video_path+" -start_number 0 -qscale:v 1 -qmin 1 -frames:v  "+str(number_of_frames)+" "+dest_folder+"/frame_%05d.jpg"

def seq_to_video_jpg(filename,out_file,frame_rate):
    out_file = out_file+"/"+filename.split('/')[-1]
    os.system("ffmpeg -r "+str(frame_rate)+" -start_number 0 -i "+filename+"/*%010d.jpg -vcodec mpeg4 -qscale:v 1 -qmin 1 -y "+out_file+".mp4")
    #x = "ffmpeg -ss "+str(start_idx)+" -i "+video_path+" -start_number 0 -qscale:v 1 -qmin 1 -frames:v  "+str(number_of_frames)+" "+dest_folder+"/frame_%05d.jpg"

def seq_to_video_modified (input_directory,out_file,frame_rate):
    import os
    os.makedirs(out_file, exist_ok= True)
    out_file = out_file + "/" + input_directory.split('/')[-1]
    x = "ffmpeg -y -pattern_type glob -r " + str(frame_rate)+ " -i '" +input_directory + "*/*.png' -vcodec mpeg4 -qmin 1 -qscale:v 1 " + out_file + ".mp4"
    os.system(x)
    print(x)

def seq_to_video_modified_vis (input_directory,frame_rate):
    import os

    out_file = input_directory + "/mask_frames"
    x = "ffmpeg -y -pattern_type glob -r " + str(frame_rate)+ " -i '" +input_directory + "/mask_frames/*.png' -vcodec mpeg4 -qmin 1 -qscale:v 1 " + out_file + ".mp4"
    os.system(x)
    out_file = input_directory + "/overlayd_frames"
    x = "ffmpeg -y -pattern_type glob -r " + str(frame_rate)+ " -i '" +input_directory + "/overlayd_frames/*.png' -vcodec mpeg4 -qmin 1 -qscale:v 1 " + out_file + ".mp4"
    os.system(x)
    out_file = input_directory + "/original_frames"
    x = "ffmpeg -y -pattern_type glob -r " + str(frame_rate)+ " -i '" +input_directory + "/original_frames/*.png' -vcodec mpeg4 -qmin 1 -qscale:v 1 " + out_file + ".mp4"
    os.system(x)
    print(x)

def all_seq_to_videos(filename,out,frame_rate):
    for subdir, dirs, files in sorted(os.walk(filename)):
        if subdir != filename:
            seq_to_video(subdir,out,frame_rate)

def all_seq_to_videos_stage1(video,filename,out,frame_rate):
    import glob
    for subdir, dirs, files in sorted(os.walk(filename)):
        #print(subdir, len(subdir.split("/")))
        #print(filename, len(filename.split("/")))
        if not subdir.__contains__("idea") and len(subdir.split("/")) == len(filename.split("/")) + 2: #see the child dirs
            print(subdir)
            seq_name = subdir.split("/")[-1] #seq name
            s = sorted(glob.glob(subdir+ "/*/*.jpg"))
            start_f = s[0].split("/")[-1]
            stop_f = s[-1].split("/")[-1]
            diff = int(stop_f.split("_")[-1][:-4]) - int(start_f.split("_")[-1][:-4])
            generate_images_n_V2_index2(video, seq_name,out,start_f,diff+1)

def seq_to_videos_stage1(video,filename,start,diff,out,frame_rate,obj_list,action_list):
    import glob
    seq_name = filename.split("/")[-1] #seq name
    #s = sorted(glob.glob(filename+ "/*/*.jpg"))
    #start_f = s[0].split("/")[-1]
    #stop_f = s[-1].split("/")[-1]
    #diff = int(stop_f.split("_")[-1][:-4]) - int(start_f.split("_")[-1][:-4])
    generate_images_n_V2_with_labels(video, seq_name,out,start,diff+1,obj_list,action_list)

def generate_N_sequence_of_actions_non_uniform(video_path,id,json_file,vid, csv_info_file,number_of_seq,output_dir,actions_per_sequence=3,upcoming_action_probgation =2,prev_action_probgation=0):
    import pandas as pd
    filter_hands_objects(json_file) # update the file to include both left and right hands
    df = pd.read_csv(csv_info_file)
    vid = video_path.split("/")[-1].split('.')[0]
    frame_rate = 60 if len(vid.split('_')[-1]) == 2 else 50
    print("Video name: ", vid, " with ", frame_rate, "fps")
    df = df[df.video_id == vid]  # filter the dataset to just include the video we're targeting
    df = df[["video_id", "start_frame", "stop_frame", "narration"]].dropna() #select the important colunms for this task
    df = df.sort_values(by=['stop_frame'])
    total_objects = df.start_frame.count()
    i = 0
    sequence_limits = []
    count_zeros=0
    sum_non_z=0
    row=[]
    sum_duration = 0
    total_files_per_folder=0
    folder_index = 1
    sum_non_z_c=0
    all_number_of_removed_images=0
    output_dir_part = id+'_'+vid+'_Part_' + str(folder_index).zfill(3)
    os.mkdir(os.path.join(output_dir,output_dir_part))
    prev_objects = []  # will store the current objects for the upcoming object list
    manual_edits=0
    while not df.empty:
        seq = df.head(actions_per_sequence)
        seqt = df.head(actions_per_sequence + upcoming_action_probgation)
        #print(df.head(9))
        min_value = seq.start_frame.min()
        max_value = seq.stop_frame.max()
        row= [min_value,max_value]
        sequence_limits.append([min_value,max_value])
        #print(seq, "\nMIN, MAX : ",min_value, max_value)
        #print("S" + str(i) + "=> " + str(min_value) + ": " + str(max_value))
        df = df[(df.start_frame > max_value)]

        if ((row[1]-row[0]) <= 0):
            print("ERROR!!, start frame of seq = ", row[0], "and end =", row[1])
            exit(1)

        #generate_images_n_V2('/Users/ahmaddarkhalil/Downloads/P30_107.mp4', './P30_107_n/', 'frame_0000000001.jpg', 3)

        start=row[0]
        stop=row[1]
        print("From prev. ",prev_objects)
        objects_set = set(prev_objects)#set(prev_objects) #will store the current objects for the upcoming object list (eidited: it was set())
        # Opening JSON file
        f = open(json_file)
        # returns JSON object as a dictionary
        data = json.load(f)
        #for propagating the actions objects

        coming_objects = [] #will store the upcoming objects for current object list
        prev_objects = []  # will store the current objects for the upcoming object list
        manual_objects=set()

        list_feq = []  # for storing the freq of the actions
        main_objects = set()
        if (len(seq) == actions_per_sequence):
            for action_probgation_index in range(0, actions_per_sequence):
                objs, frq = find_objects_in_action(data, seqt.iloc[action_probgation_index])
                list_feq.append(frq)
                for frq_key in frq.keys():
                    if frq[frq_key]  >= 4:
                        main_objects.add(frq_key)
                objs = find_objects_in_action_manually(data, seqt.iloc[action_probgation_index])
                # manual objects
                manual_objects.update(objs)

        if (len(seqt) == actions_per_sequence + upcoming_action_probgation):
            #additonal objects    
            #chosen_frq_objects = find_objects_with_usual_occurance(list_of_freq=list_feq, orignal_threshould_vote=4,number_of_annotator_votes=3,number_of_agreed_actions=2)
            chosen_frq_objects = find_objects_with_usual_occurance_v2(list_of_freq=list_feq, orignal_threshould_vote=4, total_object_count=7)
            for action_probgation_index in range (0,upcoming_action_probgation):
                objs,frq = find_objects_in_action(data, seqt.iloc[actions_per_sequence + action_probgation_index])

                #objs = choose_from_objects(objs,list_feq) #if the objects are part of the current objects with any count (just add if the current action has the object)
                objs = choose_from_objects_v2(objs, list_feq, total_object_count=2)

                coming_objects.extend(objs)

            for action_probgation_index in range(0, prev_action_probgation):
                objs, frq = find_objects_in_action(data, seqt.iloc[actions_per_sequence - action_probgation_index -1])

                #objs = choose_from_objects(objs,list_feq) #if the objects are part of the current objects with any count (just add if the current action has the object)
                objs = choose_from_objects_v2(objs, list_feq, total_object_count=2)

                prev_objects.extend(objs)

        # Iterating through the json list
        for folder_name in data.keys():
            frame_number = int(folder_name.split('/')[-1].split('_')[-1])
            if(frame_number >= start and frame_number <= stop):
                objects = data[folder_name]
                #print("ADD ",objects_set)
                objects_set.update(set(objects["labels"]))

        print("Final Set before adding", objects_set)
        print("Additional added objects (vote):",chosen_frq_objects)
        objects_set.update(set(chosen_frq_objects))
        print("From upcoming: ",coming_objects)
        #objects_set.update(set(coming_objects)) ## importatn!!, initailize the set of object with the upcoming acitons objects

        if (not manual_objects.issubset(objects_set)):
            manual_edits = manual_edits+1

        print("Manual detected objects:",manual_objects)
        objects_set.update(manual_objects)

        objects_set_new = delete_duplicates(objects_set)
        print("Deleted objects: ", objects_set - objects_set_new)
        objects_set = objects_set_new

        print("Final Set after adding", objects_set)
        print("Total manual edits is:",manual_edits)


        if(objects_set.__len__() == 0):
            count_zeros = count_zeros + 1
        else:
            sum_non_z = sum_non_z + objects_set.__len__()

            sum_duration =sum_duration + (row[1]-row[0] + 1)/frame_rate # there we calculate the duration of an action, +1 since all inclusive,
            print("Time of the sequence: ", (row[1]-row[0] + 1)/frame_rate)
            print("S" + str(i+1) + "=> " + str(row[0]) + ": " + str(row[1]))
            output_name = id+'_'+vid+"_seq_"+str(i+1).zfill(5)
            print(output_name)
            # Path
            path = os.path.join(output_dir+output_dir_part,output_name)
            if not os.path.exists(path):
                os.mkdir(path,mode=0o777)
            frame_name = id+'_'+'frame_'+str(start).zfill(10)+'.jpg'
            print(frame_name)

            all_selected_frames = []
            for index_seq, row_seq in seq.iterrows():
                print("Start, end: ", row_seq['start_frame'], row_seq['stop_frame'])
                ##there
                selected_frames = select_frames_of_action(video_path,row_seq['start_frame'], row_seq['stop_frame'],4,10, 10) # divide the action into 4 divisions to select frames
                all_selected_frames.extend(selected_frames)
                print("Selected: ", selected_frames)

            all_selected_frames = sorted(all_selected_frames)
            all_selected_frames_updated=[]
            number_of_removed_images = 0

            ii=0
            while ii < len(all_selected_frames) - 1: #remove elements that are close
                all_selected_frames_updated.append(all_selected_frames[ii]) #store the value into new array
                flag = 0
                k= ii + 1
                while (k < (len(all_selected_frames) - 1) and abs(all_selected_frames[ii] - all_selected_frames[k]) < 25): #if they have less or equal to 25 frames gap, then remove the second point
                    print("Deleted point", all_selected_frames[ii],"," ,all_selected_frames[k])
                    number_of_removed_images = number_of_removed_images + 1
                    k = k + 1 # skip the next iteration
                    flag = 1

                if flag == 0: # go to next element
                    ii = ii + 1
                else:
                    ii=k #start from the point the loop stoped

            if (abs(all_selected_frames[ii] - all_selected_frames[ii-1]) > 25):
                all_selected_frames_updated.append(all_selected_frames[ii])
            else:
                number_of_removed_images = number_of_removed_images + 1
            print("seq all:",all_selected_frames)
            print("seq selected",all_selected_frames_updated)
            try:
                all_selected_frames_updated = resample_frames(all_selected_frames_updated,number_of_removed_images)
                all_selected_frames_updated = sorted(all_selected_frames_updated)
            except:
                all_selected_frames_updated = []
                print("LAST ACTION EXCLUDED")
            if len(all_selected_frames_updated) == 6:
                all_number_of_removed_images = all_number_of_removed_images + number_of_removed_images
                for selected_frame in all_selected_frames_updated:
                        frame_name = id+"_"+vid+'_'+'frame_' + str(selected_frame).zfill(10) + '.jpg'
                        image_store_loc = os.path.join(path, id+'_'+'frame_' + str(selected_frame).zfill(10))
                        #this is to test if the image is exits or not, in the overalpping cases of actions
                        try:
                            os.mkdir(image_store_loc)  # to create folder for each image
                            sum_non_z_c = sum_non_z_c + 1
                            total_files_per_folder = total_files_per_folder + 1
                        except:
                            print("Error!! duplicate element!! check the if statement in the all_selected_frames array" )
                            continue
                        ##there
                        generate_images_n_V2_index1(video_path, image_store_loc, frame_name, frame_rate) #generate the image
                        new_frame_name = id+"_"+vid+'_'+'frame_' + str(selected_frame).zfill(
                            10) + '.jpg'
                        ##there
                        os.rename(os.path.join(image_store_loc, "frame_00001.jpg"), os.path.join(image_store_loc, new_frame_name))

                        if ("other cutlery" in objects_set):
                            objects_set.remove("other cutlery")

                        m = {"labels": list(objects_set)}
                        m2 = {"main_objects": list(main_objects)}

                        import collections, functools, operator
                        m_freq = set()
                        if (len(list_feq) != 0):
                            m_freq = dict(functools.reduce(operator.add,
                                                       map(collections.Counter, list_feq)))
                        m_count = {"labels_counts": (m_freq)}
                        global set_of_jsons
                        set_of_jsons[os.path.join(image_store_loc, new_frame_name)] = m
                        #set_of_jsons[os.path.join(image_store_loc, new_frame_name)].update(m2)
                        #set_of_jsons[os.path.join(image_store_loc, new_frame_name)].update(m_count)

                seq_to_videos_stage1(video_path, path,str(row[0])+".jpg",row[1]-row[0], 'video_seq_n', 50,list(objects_set),seq.narration)

                i = i + 1
                #if(i == number_of_seq): # for breaking on a certain amount of seq.
                #    break
                if (total_files_per_folder > 180):
                    folder_index = folder_index + 1
                    output_dir_part = id+'_'+vid+'_Part_' + str(folder_index).zfill(3)
                    output_json_part = id+'_'+vid+'_Part_' + str(folder_index-1).zfill(3)
                    file = open(output_json_part+".json", "w")
                    a = json.dumps(set_of_jsons)
                    file.write(str(a))
                    file.close()
                    set_of_jsons = {}
                    os.mkdir(os.path.join(output_dir, output_dir_part))

                    print("Number of files in this folder: ",total_files_per_folder)
                    total_files_per_folder = 0
    if (total_files_per_folder > 0):
        output_json_part = id+'_'+vid+'_Part_' + str(folder_index).zfill(3)
        file = open(output_json_part + ".json", "w")
        a = json.dumps(set_of_jsons)
        file.write(str(a))
        file.close()


    average_seq_duration = sum_duration/len(sequence_limits)
    sum_of_all_seq_frames = round(sum_duration*frame_rate,0)

    print("Average duration per sequence: ",average_seq_duration, "seconds")
    print("Number of all annotations: ",sum_non_z_c)
    print("Number of appended frames (fix the number to 6 f/seq):",all_number_of_removed_images)
    print("Number of sequences: ", i)
    print("Json file length = ", len(data)/4)
    print("Number of actions in the csv file: ",total_objects)
    print("Number of missing actions within sequence in json file: ", count_zeros)
    print("Number of  objects: ", sum_non_z)
    print("Random value difference:",tr)


def delete_duplicates(objects_set):
    import re
    mylist = list(objects_set)
    newlist = set()
    for x in mylist:
        r = re.compile(".*\s+" + x + "$")
        result = list(filter(r.match, mylist))  # Read Note below
        if len(result) == 0:
            newlist.add(x)
        #print(x, " =>> ", newlist)

    #print("Final list:")
    #print(newlist)
    return newlist
def choose_from_objects(objs,list_feq): #this to filter the upcoming, prev objects according the current list of object counts
    objects = set()
    for obj in objs:
        for x in list_feq:
            if(obj in x.keys()):
                objects.add(obj)
                break
    return objects

def choose_from_objects_v2(objs,list_feq,total_object_count): #this to filter the upcoming, prev objects according the current list of object counts
    import collections, functools, operator
    objects_counts = []
    for obj in objs:
        for x in list_feq:
            if(obj in x.keys()):
                objects_counts.append({obj:x[obj]})

    # sum the values with same keys
    if objects_counts == []:
        return []

    result = dict(functools.reduce(operator.add,
                                   map(collections.Counter, objects_counts)))
    objects_set = set()
    for k2 in result.keys():
        if (result[k2] >= total_object_count):
            objects_set.add(k2)
    return objects_set
def find_objects_in_action(data, action):
    #find all actions with this name (narration)
    possible_actions = {k:v for k, v in data.items() if action.narration.replace(" ", "_") in k}
    objects_set = set()
    objects_freq = {}
    # Iterating through the list of possible actions
    for folder_name in possible_actions:
        frame_number = int(folder_name.split('/')[-1].split('_')[-1])
        if (frame_number >= action.start_frame and frame_number <= action.stop_frame):
            objects = data[folder_name]
            # print("ADD ",objects_set)
            #for object in objects["labels"]:
            #    objects_set.add(object)
            #
            return objects["labels"],objects["labels_freq"]
    return objects_set,objects_freq

def find_objects_in_action_manually(data, action):
    #find all actions with this name (narration)
    possible_actions = {k:v for k, v in data.items() if action.narration.replace(" ", "_") in k}
    objects_set = set()
    objects_freq = {}

    for folder_name in possible_actions:
        frame_number = int(folder_name.split('/')[-1].split('_')[-1])
        if (frame_number >= action.start_frame and frame_number <= action.stop_frame):
            objects = data[folder_name]

            # STEP 1: manual checks for narration(or verb) => objects ###############################################
            if "chop" in folder_name or "cut" in folder_name:
                objects_set.add("chopping board")
                objects_set.add("knife")
            if "wash" in folder_name or "rinse" in folder_name:
                objects_set.add("sink")
                objects_set.add("tap")
            if "stir" in folder_name:
                objects_set.add("spoon")

            # ٍSTEP 2: check if the objects are part of the narration, if so, it would be added to object list########
            for k2 in objects["labels_freq"].keys():
                if k2.replace(" ","_") in folder_name and "hand" not in k2:
                    objects_set.add(k2)

            #STEP 3: manual checks for object => object
            for k3 in objects["labels"]:
                if k3.replace(" ", "_") == "extractor":
                    objects_set.add("hob")
               #if k3.replace(" ", "_") == "chopping_board":
               #     objects_set.add("knife")
               # if k3.replace(" ", "_") == "knife":
               #     objects_set.add("chopping board")
                if k3.replace(" ", "_") == "sink":
                    objects_set.add("tap")
                if k3.replace(" ", "_") == "tap":
                    objects_set.add("sink")
                if k3.replace(" ", "_") == "colander":
                    objects_set.add("tap")
                    objects_set.add("sink")

    return objects_set


def find_objects_with_usual_occurance(list_of_freq, orignal_threshould_vote,number_of_annotator_votes,number_of_agreed_actions): #this for objects that accures in more than one action to consider
    from collections import Counter
    objects_list = []
    for index in range (0,len(list_of_freq)):
        for key in list_of_freq[index].keys():
            if (list_of_freq[index][key] >= number_of_annotator_votes and list_of_freq[index][key] < orignal_threshould_vote):
                objects_list.append(key)

    dict_of_objects = Counter(objects_list)
    objects_set = set()
    for k2 in dict_of_objects.keys():
        if (dict_of_objects[k2] >= number_of_agreed_actions):
            objects_set.add(k2)
    return list(objects_set)

def find_objects_with_usual_occurance_v2(list_of_freq, orignal_threshould_vote,total_object_count): #this for objects that accures in more than one action to consider
    #there we just see the overall count of each object
    import collections, functools, operator
    objects_counts = []
    for index in range (0,len(list_of_freq)):
        for key in list_of_freq[index].keys():
            if (list_of_freq[index][key] < orignal_threshould_vote):
                objects_counts.append({key:list_of_freq[index][key]})

    if objects_counts == []:
        return []

    # sum the values with same keys
    result = dict(functools.reduce(operator.add,
                                   map(collections.Counter, objects_counts)))
    objects_set = set()
    for k2 in result.keys():
        if (result[k2] >= total_object_count):
            objects_set.add(k2)
    return list(objects_set)

def  resample_frames(all_selected_frames_updated,number_of_removed_images):
    for index_r in range(0,number_of_removed_images):
        all_selected_frames_updated = sorted(all_selected_frames_updated)
        selected_frame = find_sampled_frame(all_selected_frames_updated)
        all_selected_frames_updated.append(selected_frame)
    return all_selected_frames_updated

def find_sampled_frame(list_of_frames):
    selected_index=0
    max_dist=0
    for frame_index in range (0,len(list_of_frames) - 1):
        dist = abs(list_of_frames[frame_index] - list_of_frames[frame_index+1])
        if (dist > max_dist):
            max_dist = dist
            selected_index = frame_index
    return int((list_of_frames[selected_index] + list_of_frames[selected_index+1])/2)

def select_frames_of_action(video_path,start, stop, number_of_divisions,random_ratio_value,blur_window):
    """
    this function decide which frames to take from start to stop of the action,
    "number_of_division" determain how many dividions the action would be divided,
    we'll take the start of second division (end of first division)and start of final division (end of the previous division) as selected frames with offest for motion blur
    """
    import os
    import random
    global tr
    selected_frames = []
    ratio = int ((stop - start) * 1 / number_of_divisions)
    random_ratio = int(random.randrange(-int((stop - start)), int((stop - start)) + 1) / random_ratio_value) # e.g 10% random value of the frame
    tr= tr + random_ratio
    start_frame = start + ratio + random_ratio

    random_ratio = int(random.randrange(-int((stop - start)), int((stop - start)) + 1) / random_ratio_value)
    tr= tr + random_ratio
    stop_frame = stop - ratio + random_ratio

    selected_frames.append(start_frame + select_lowest_blur(video_path, start_frame, blur_window)) #add the first selected frame,
    selected_frames.append(stop_frame + select_lowest_blur(video_path, stop_frame, blur_window)) #add the second selected frame,
    return selected_frames



def select_lowest_blur (video_path, start_frame, blur_window):
    import shutil
    image_store_loc = "temp" #temp directory
    os.makedirs(image_store_loc, exist_ok= True)
    generate_images_n_V3_index1(video_path, image_store_loc, start_frame - blur_window, 2*blur_window + 1)
    index=0
    max_blur = 0 #small value
    max_index = - blur_window # since it + or - blur_window
    for i in range(-blur_window ,blur_window):
        image_path = os.path.join(image_store_loc,'{0}'.format(str(i + blur_window + 1).zfill(5))+".jpg")
        blur_value = measure_blur_image(image_path)
        if blur_value > max_blur:
            max_blur = blur_value
            max_index = i

    shutil.rmtree(image_store_loc)
    return max_index

def json_to_csv_stage1(jsonfile, csvfile):
    import json
    import csv
    import pandas as pd
    f = open(jsonfile)
    fcsv = open(csvfile, "w")

    # returns JSON object as a dictionary
    data = json.load(f)
    max_objects=0
    # Iterating through the json list
    for folder_name in data.keys():
        objects = data[folder_name]["labels"]

        if len(objects) > max_objects:
            max_objects = len(objects)
        print(folder_name, " has ",objects)


        #for object in objects["labels"]:
            #objects_set.add(object)
        print("Max number of objects: ", max_objects)
    header="Frame"
    #    for i in range (1, max_objects + 1):
    #    header = header + ",object " + str(i)
    #fcsv.write(header)
    #fcsv.write('\n')

    for folder_name in data.keys():
        objects = data[folder_name]["labels"]
        row = folder_name
        row = row + ","+",".join(objects)
        fcsv.write(row)
        fcsv.write('\n')

    f.close()
    fcsv.close()

def csv_to_json_stage1(csvfile,jsonfile):
    import json
    import csv
    import pandas as pd

    set_of_files = {}

    f = open(jsonfile, "w")
    fcsv = open(csvfile, "r")
    reader = csv.reader(fcsv, delimiter=',')


    for row in reader:
        index = 0
        objects_list = []
        key=""
        for column in row:
            if(index == 0):
                if (column == ""):
                    print("ERROR missing action name, check the end of the file!!!")
                    exit(1)

                print("DONE: ", column)
                key = column
            else:
                if(column != ""):
                    objects_list.append(column)
            index = index + 1
            m = {"labels": objects_list}
            set_of_files[key] = m

    a = json.dumps(set_of_files)
    f.write(str(a))  # delete the first and last char (array []) as required in the json file

    f.close()
    fcsv.close()
    print ("DONE!!")

def merge_actions_objects(json_file, csv_info_file,vid, number_of_merged_actions):
    import pandas as pd
    df = pd.read_csv(csv_info_file)
    frame_rate = 60 if len(vid.split('_')[-1]) == 2 else 50
    print("Video name: ", vid)
    df = df[df.video_id == vid]  # filter the dataset to just include the video we're targeting
    df = df[["video_id", "start_frame", "stop_frame"]].dropna() #select the important colunms for this task
    df = df.sort_values(by=['stop_frame'])
    total_objects = df.start_frame.count()

def filter_annotators (json_file, number_of_annotators, vote_number,outputFile): #vote number is for min number of object vote to be selected
    import json
    from collections import Counter
    data = json.load(open(json_file))
    all_data = {}
    dict_of_objects = {}
    objects_set = set()
    prev_seq=""
    number_of_objects=0
    for k in (data.keys()):
        seq = k.split("/")[1]
        if (seq != prev_seq):
            list_of_objects = []
            objects_set = set()
            for index in range (0,number_of_annotators):
                list_of_objects.extend(data[k]["label_"+str(index)])
            dict_of_objects = Counter(list_of_objects)
            for k2 in dict_of_objects.keys():
                if (dict_of_objects[k2] >= vote_number):
                    objects_set.add(k2)
                    number_of_objects=number_of_objects+1
            prev_seq = seq
        objects = {"labels": list(objects_set)}
        frequenceies = {"labels_freq": dict_of_objects}

        all_data[k] = (frequenceies)
        all_data[k].update(objects)
    #dump it
    json.dump(all_data,open(outputFile,"w"))
    print("Total number of objects:",number_of_objects)

def delete_overlap_csv(csv_file,vid):
    import pandas as pd
    to_export = []
    df = pd.read_csv(csv_file)
    print("Video name: ", vid)
    df = df[df.video_id == vid]  # filter the dataset to just include the video we're targeting
    #df = df[["video_id", "start_frame", "stop_frame", "narration"]].dropna() #select the important colunms for this task
    df = df.sort_values(by=['start_frame'])
    index  = 0
    while not df.empty:
        action = df.head(1)
        to_export.append(action)
        #print(action.narration.to_string())
        index = index + 1
        df = df[(df.start_frame > int(action.stop_frame))]
    to_export = pd.concat(to_export)
    #to_export.to_csv('EPIC_100_train_new.csv',index=False)
    print("Total actions:",index)
    return to_export
def delete_overlap_csv_all_videos(csv_file):
    import pandas as pd
    to_export = []
    df = pd.read_csv(csv_file)
    df = df.sort_values(by=['video_id'])
    videos = pd.unique(df.video_id)
    for i in videos:
        to_export.append(delete_overlap_csv(csv_file,i))
    to_export = pd.concat(to_export)
    #to_export.to_csv('EPIC_100_train_new.csv',index=False)
    to_export.to_csv("nonoverlap_"+csv_file, index=False)

def fill_objects_gaps(json_file, max_gap=1): #this function to fill the gaps between sequences (e.g. if s1 and s3 have the object and s2 does not have, add the object to s2)
    import json
    data = json.load(open(json_file))
    all_data = {}
    all_data_keys={}
    per_seq_data={}
    main_objects = set()
    dict_of_objects = {}
    objects_set = set()
    prev_seq = ""
    number_of_objects = 0
    for k in (data.keys()):
        seq = k.split("/")[2]
        if (seq != prev_seq):
            #print(seq, " >>> ", data[k]["labels"])
            per_seq_data[seq] = {"labels":data[k]["labels"],"labels_counts":data[k]["labels_counts"],"main_objects":data[k]["main_objects"]}
        prev_seq = seq
    keys = sorted(per_seq_data.keys())
    edits = 0
    for i, _ in enumerate(sorted(keys)):
        main_objects.update(per_seq_data[keys[i]]["main_objects"])
        for obj in main_objects:
            if obj in per_seq_data[keys[i]]["labels_counts"] and per_seq_data[keys[i]]["labels_counts"][obj] >= 4:
                per_seq_data[keys[i]]["labels"].append(obj)

    for i,_ in enumerate(sorted(keys)):
        print(keys[i], " >>before adding>> ", per_seq_data[keys[i]]["labels"])
        seq_set = set()
        seq_set_temp = set()
        f=0


        if (i >= max_gap and i < len(keys) - max_gap):
            for boundary_boject in intersection(per_seq_data[keys[i-1]]["labels"],per_seq_data[keys[i+1]]["labels"]): #this find the intersection of the prev and next seq
                seq_set.add(boundary_boject)
            if not (seq_set <= set(per_seq_data[keys[i]]["labels"])): # to check if the seq_set is part of labels
                seq_set_temp = set(per_seq_data[keys[i]]["labels"]) #just for printing
                seq_set.update(set(per_seq_data[keys[i]]["labels"]))
                per_seq_data[keys[i]]["labels"] = list(seq_set)
                edits = edits + 1
                f=1
            if f== 1:
                print(keys[i - 1], " >>prev>> ", per_seq_data[keys[i - 1]]["labels"])
                print(keys[i+1], " >>next>> ", per_seq_data[keys[i+1]]["labels"])
                print(keys[i],">>Added items>>: ", seq_set - seq_set_temp)
                print(keys[i], " >>after adding>> ", per_seq_data[keys[i]]["labels"])


        if ("other cutlery" in per_seq_data[keys[i]]["labels"]):
            per_seq_data[keys[i]]["labels"].remove("other cutlery")
            print ("Other is detected!!")
        print(keys[i], " >>after adding>> ", set(per_seq_data[keys[i]]["labels"]))
        print(keys[i], " >>after adding>> ", len(per_seq_data[keys[i]]["labels"]))
        all_data_keys[keys[i]] = ({"labels":list(set(per_seq_data[keys[i]]["labels"]))})

    for k in (data.keys()):
        seq = k.split("/")[2]
        if (seq in all_data_keys.keys()):
            all_data[k] = all_data_keys[seq]

    print("\n\nTotal edits = ",edits)
    #dump it
    json.dump(all_data,open("gaps_"+json_file,"w"))

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def fill_objects_gaps_f(json_file): #this function to fill the gaps between sequences (e.g. if s1 and s3 have the object and s2 does not have, add the object to s2)
    import json
    data = json.load(open(json_file))
    all_data = {}
    all_data_keys={}
    per_seq_data={}
    main_objects = set()
    dict_of_objects = {}
    objects_set = set()
    prev_seq = ""
    number_of_objects = 0
    for k in (data.keys()):
            all_data[k] = {"labels":data[k]["labels"]}

    #dump it
    json.dump(all_data,open("gaps_"+json_file,"w"))


def do_stats_stage1_jsons(folder):
    import json
    import glob
    import collections, functools, operator
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    from PIL import Image
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    total_repos=0
    total_number_of_images=0
    total_number_of_objects = 0
    total_number_of_seq = 0
    total_number_objects_per_image=[]
    objects=[]
    objects_per_part = []
    prev_file=""
    for infile in sorted(glob.glob(folder + '/*.json')):

        if (infile.split("/")[-1].split("_")[0] != prev_file):
            #print("From ", prev_file + " ==> to ", infile)
            objects_per_part = []
        #else:
            #print("Contiune: ",infile)

        prev_file = infile.split("/")[-1].split("_")[0]

        f = open(infile, 'r')
        annotation_info = json.load(f)
        #print(f"Stage1: {infile.split('/')[-1]} ==> {len(annotation_info)}")
        total_repos = total_repos + 1
        total_number_of_images = total_number_of_images + len(annotation_info)

        prev_seq = ""
        for k in sorted(annotation_info.keys()):
            seq = k.split("/")[2]
            if (seq != prev_seq):
                total_number_of_seq = total_number_of_seq + 1
                prev_seq = seq
            total_number_of_objects = total_number_of_objects + len(annotation_info[k]["labels"]) #add each image objects to the list of objects
            total_number_objects_per_image.append(len(annotation_info[k]["labels"]))
            objects.extend(annotation_info[k]["labels"])
            objects_per_part.extend(annotation_info[k]["labels"])
        #print("Number of unique objects: ", len(set(objects_per_part)))
        #print("Number of all objects: ", len((objects_per_part)))

    import pandas as pd
    objs = set(objects)
    dft = pd.DataFrame(objs)
    dft.to_csv('objects.csv')
    objects_counts = collections.Counter(objects)


    df = pd.DataFrame.from_dict(objects_counts, orient='index').reset_index()
    #print(df)
    print("STAGE1 stats")
    print("Number of repos: ",(total_repos))
    print("Number of sequences: ", (total_number_of_seq))
    print("Number of images (masks): ", (total_number_of_images))
    print("Number of objects: ", (total_number_of_objects))
    print("Number of unique objects: ", len(set(objects)))
    print("Most common objects: ")
    df = pd.DataFrame(objects_counts.most_common(5), columns=["Object", "Image count"])
    print(df)

    # Fit a normal distribution to
    # the data:
    # mean and standard deviation
    mu, std = norm.fit(total_number_objects_per_image)
    print("(Min, Max, Mean, st. deviation) of the number of objects image and per sequence (stage 1): (",min(total_number_objects_per_image),",",
          max(total_number_objects_per_image),",",round(mu,2),",",round(std,2),")")
    df = pd.DataFrame(total_number_objects_per_image, columns=["Number of objects"])
    df['Stage number'] = ['Stage 1' for x in range(0, len(total_number_objects_per_image))]
    # read a tips.csv file from seaborn library
    # count plot on two categorical variable
    sns.countplot(x='Number of objects', data=df, color="salmon")
    plt.savefig("hist_stage1.png")
    #plt.show()
    return df







def do_stats_stage2_jsons(folder):

    import json
    import glob
    import collections, functools, operator
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    from PIL import Image
    from scipy.stats import norm
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    total_repos=0
    total_number_of_images=0
    total_number_of_objects = 0
    total_number_of_seq = 0
    total_number_objects_per_image=[]
    objects=[]
    for infile in sorted(glob.glob(folder + '/*.json')):
        f = open(infile)
        # returns JSON object as a dictionary
        data = json.load(f)
        #print(f"Stage2: {infile.split('/')[-1]} ==> {len(data)}")

        total_repos = total_repos + 1
        #sort based on the folder name (to guarantee to start from its first frame of each sequence)
        data = sorted(data, key=lambda k: k['documents'][0]['directory'])

        total_number_of_images = total_number_of_images + len(data)

        # Iterating through the json list
        index = 0
        full_path=""
        prev_seq = ""
        obj_per_image=0
        for datapoint in data:
            obj_per_image=0 # count number of objects per image
            seq = datapoint['documents'][0]['directory'].split("/")[2]
            if (seq != prev_seq):
                total_number_of_seq = total_number_of_seq + 1
                prev_seq = seq
            image_name = datapoint["documents"][0]["name"]
            image_path = datapoint["documents"][0]["directory"]
            masks_info = datapoint["annotation"]["annotationGroups"][0]["annotationEntities"]
            #generate_masks(image_name, image_path, masks_info, full_path) #this is for saving the same name (delete the if statemnt as well)
            entities = masks_info
            for entity in entities: #loop over each object
                object_annotations = entity["annotationBlocks"][0]["annotations"]
                if not len(object_annotations) == 0: #if there is annotation for this object, add it
                    total_number_of_objects = total_number_of_objects + 1
                    objects.append(entity["name"])
                    obj_per_image = obj_per_image + 1
            total_number_objects_per_image.append(obj_per_image)

    # these lines to save the unique objects
    df1 = pd.DataFrame(set(objects))
    # saving the dataframe
    df1.to_csv('objects_list.csv',index=False)

    objects_counts = collections.Counter(objects)

    df = pd.DataFrame.from_dict(objects_counts, orient='index').reset_index()
    #print(df)
    print("STAGE2 stats")
    print("Number of repos: ",(total_repos))
    print("Number of sequences: ", (total_number_of_seq))
    print("Number of images (masks): ", (total_number_of_images))
    print("Number of objects: ", (total_number_of_objects))
    print("Number of unique objects: ", len(set(objects)))
    print("Number of hand objects: ", objects_counts['left hand'] + objects_counts['right hand'], " Percentage:",(objects_counts['left hand'] + objects_counts['right hand'])/sum(objects_counts[k] for k in objects_counts.keys()))
    print("Number of non-hand objects: ", sum(objects_counts[k] for k in objects_counts.keys()) - (objects_counts['left hand'] + objects_counts['right hand']), " Percentage:",(sum(objects_counts[k] for k in objects_counts.keys()) - (objects_counts['left hand'] + objects_counts['right hand']))/sum(objects_counts[k] for k in objects_counts.keys()))
    print("Most common objects: ")
    df = pd.DataFrame(objects_counts.most_common(5), columns=["Object", "Image count"])
    print(df)

    # Fit a normal distribution to
    # the data:
    # mean and standard deviation
    from matplotlib import pyplot
    mu, std = norm.fit(total_number_objects_per_image)

    print("(Min, Max, Mean, st. deviation) of the number of objects image (stage 2): (",min(total_number_objects_per_image),",",
          max(total_number_objects_per_image),",",round(mu,2),",",round(std,2),")")

    import ast
    data_file = pd.read_csv('EPIC_100_noun_classes.csv')
    all_objects_strings = data_file['instances']
    #print(sum([x in list(nouns) for x in set(objects)]))
    all_objects_lists = [ast.literal_eval(x) for x in all_objects_strings] # convert the string to array (from "[]" to [])
    all_objects = sum(all_objects_lists, []) # append all lists into one big list [[],[],...[]] => []
    all_objects_filtered = set([(" ".join(x.split(":")[1:]) + " " + x.split(":")[0]).strip() for x in all_objects]) # get the original data objects
    objects_in_all_objects = [x in all_objects_filtered for x in set(objects)]
    print(f"Stage2=> Percentage of objects that are in the EPIC-KITCHENS => {sum(objects_in_all_objects)/len(objects_in_all_objects)}")
    print(f"Stage2=> Number of objects that are not in  EPIC-KITCHENS => {len(set(objects) - all_objects_filtered)} out of {len(set(objects))}")


    df = pd.DataFrame(total_number_objects_per_image, columns=["Number of objects"])
    # read a tips.csv file from seaborn library
    # count plot on two categorical variable
    sns.countplot(x='Number of objects', data=df, color="salmon")
    plt.savefig("hist_stage2.png")
    #plt.show()

    df['Stage number'] = ['Stage 2' for x in range(0, len(total_number_objects_per_image))]
    df = [do_stats_stage1_jsons('completed_annotations_stage1'),df]
    df = pd.concat(df)
    fig, ax = plt.subplots()
    # the size of A4 paper
    fig.set_size_inches(10, 6)

    ax=sns.countplot(x='Number of objects',hue='Stage number', data=df)
    plt.xlabel('Number Of Objects', fontsize=16)
    plt.ylabel('Image count', fontsize=16)
    plt.legend(loc='upper right',fontsize=14)
    plt.savefig("hist_stage1 and 2.png")

    #plt.show()
    print("Percentage of images without any object (number of objects=0): ",100*round(total_number_objects_per_image.count(0)/len(total_number_objects_per_image),3),"%")

    mask = np.array(Image.open('cloud_bg.png'))
    wordcloud = WordCloud(width = 2048, height = 2048, random_state=1, background_color='white', colormap='rainbow', collocations=False, stopwords = STOPWORDS,mask=mask)
    wordcloud.generate_from_frequencies(frequencies=dict(objects_counts.most_common(200)))
    wordcloud.to_file("wordcloud.png")

def do_stats_stage2_jsons_single_file(file):
    """_summary_

    Args:

    Returns:
        a list, with (name(str), num_occurrence)
    """

    total_number_of_images=0
    total_number_of_objects = 0
    total_number_of_seq = 0
    total_number_objects_per_image=[]
    objects=[]
    infile=file
    f = open(infile)
    # returns JSON object as a dictionary
    data = json.load(f)

    #sort based on the folder name (to guarantee to start from its first frame of each sequence)
    data = sorted(data, key=lambda k: k['documents'][0]['directory'])

    total_number_of_images = total_number_of_images + len(data)

    # Iterating through the json list
    index = 0
    full_path=""
    prev_seq = ""
    obj_per_image=0
    for datapoint in data:
        obj_per_image=0 # count number of objects per image
        seq = datapoint['documents'][0]['directory'].split("/")[2]
        if (seq != prev_seq):
            total_number_of_seq = total_number_of_seq + 1
            prev_seq = seq
        image_name = datapoint["documents"][0]["name"]
        image_path = datapoint["documents"][0]["directory"]
        masks_info = datapoint["annotation"]["annotationGroups"][0]["annotationEntities"]
        #generate_masks(image_name, image_path, masks_info, full_path) #this is for saving the same name (delete the if statemnt as well)
        entities = masks_info
        for entity in entities: #loop over each object
            object_annotations = entity["annotationBlocks"][0]["annotations"]
            if not len(object_annotations) == 0: #if there is annotation for this object, add it
                total_number_of_objects = total_number_of_objects + 1
                objects.append(entity["name"])
                obj_per_image = obj_per_image + 1
        total_number_objects_per_image.append(obj_per_image)

    objects_counts = collections.Counter(objects)
    import pandas as pd

    df = pd.DataFrame.from_dict(objects_counts, orient='index').reset_index()
    #print(df)
    #print("Number of sequences: ", (total_number_of_seq))
    #print("Number of images (masks): ", (total_number_of_images))
    #print("Number of unique objects: ", len(set(objects)))

    return objects_counts.most_common()

def check_frame_rate(folder):
    import glob
    import json
    x = []
    duration = 0
    number_of_annotated_images = 0
    fr_per_seq = []
    fps=50
    total_repos = 0
    total_number_of_images = 0
    total_number_of_objects = 0
    total_number_of_seq = 0
    total_number_objects_per_image = []
    objects = []
    objects_per_part = []
    prev_file = ""
    sum_fps = 0
    sum_fps2 = 0
    max_action_diff = []
    seq_duration = []
    for infile in sorted(glob.glob(folder + '/*.json')):

        if (infile.split("/")[-1].split("_")[0] != prev_file):
            # print("From ", prev_file + " ==> to ", infile)
            objects_per_part = []
        # else:
        # print("Contiune: ",infile)

        prev_file = infile.split("/")[-1].split("_")[0]

        f = open(infile, 'r')
        annotation_info = json.load(f)
        # print(f"Stage1: {infile.split('/')[-1]} ==> {len(annotation_info)}")
        total_repos = total_repos + 1
        total_number_of_images = total_number_of_images + len(annotation_info)

        prev_seq = ""
        prev_frame = 0
        total_duration = 0 #at the end of each seq, it will be calculated
        f_frame = 0
        diff = 1 #it will be calculated between each 2 frames
        i=0
        img_per_seq=1
        first_image = 0
        last_image=0
        max_action=0
        for k in sorted(annotation_info.keys()):
            i = i+1
            seq = k.split("/")[2]
            frame = int(k.split('/')[-1][-14:-4])
            if i == 1:
                first_image = frame

            if (seq != prev_seq or i == len(annotation_info)):
                #print(f"seq = {prev_seq}")
                prev_seq = seq
                if i == len(annotation_info):
                    #print(f"from {frame} to {f_frame}")
                    img_per_seq = img_per_seq + 1
                    total_duration = (total_duration + frame - f_frame+1)
                    seq_duration.append(frame - f_frame+1)
                    diff = diff + (frame - prev_frame)
                    if (frame - prev_frame) > max_action:
                        max_action = (frame - prev_frame)
                    max_action_diff.append(max_action)
                    max_action = 0
                    fr_per_seq.append((fps*img_per_seq)/(frame - f_frame+1))
                    #print(f"total = {total_duration}")
                    #print(f"diff = {diff}")
                    last_image = frame
                    assert (diff == total_duration)
                else:
                    total_number_of_seq = total_number_of_seq + 1
                    #print(f"from {prev_frame} to {f_frame}")
                    total_duration = (total_duration + prev_frame - f_frame+1)
                    if total_duration != 0 and img_per_seq!= 1:
                        fr_per_seq.append((fps * img_per_seq) / (prev_frame - f_frame+1))
                        seq_duration.append(prev_frame - f_frame+1)
                        max_action_diff.append(max_action)
                        max_action = 0
                #print(f"total = {total_duration}")
                #print(f"diff = {diff}")
                prev_frame = 0
                #print("img/seq", img_per_seq)
                img_per_seq = 1
                f_frame = frame
            if prev_frame != 0:
                diff = diff + (frame - prev_frame)
                img_per_seq = img_per_seq + 1
                if (frame - prev_frame) > max_action:
                    max_action = (frame - prev_frame)
                #print(f"{frame} - {prev_frame} => diff = {diff}")
            else:
                diff = diff + 1

            prev_frame = frame
        #sum_fps = sum_fps + (50*len(annotation_info))/(total_duration)
        sum_fps = sum_fps + diff -1 # (-1 to cancel the final addition of the frames)
        sum_fps2 = sum_fps2 + total_duration
        print(f"Repo frame rate for {infile.split('/')[-1]} is {(fps*len(annotation_info))/(last_image-first_image + 1)} fps")
        x.append((fps*len(annotation_info))/(last_image-first_image + 1))
        duration = duration + (last_image-first_image + 1)
        number_of_annotated_images = number_of_annotated_images + len(annotation_info)
    print(f"Number of repos: {total_repos}")
    print(f"Number of sequences: {total_number_of_seq}")
    print(f"Number of annotated frames: {total_number_of_images}")
    print(f"Total number of frames {sum_fps}")
    #print(f"Total: per sequence sum = {sum_fps2}")

    print("Percentage of annotated frames : ",round(total_number_of_images/sum_fps,4)*100, " %")
    print("Annotation rate (calculated per image): ",round(total_number_of_images/sum_fps,4)*fps , " fps")
    print("Annotation rate (calculated per sequence): ", sum(fr_per_seq)/len(fr_per_seq), " fps")
    print("Annotation rate (calculated per repo): ",sum(x)/len(x), " fps")
    print(len(fr_per_seq))

    max_action_rate = [i / j for i, j in zip(max_action_diff, seq_duration)]
    print("Percentage of the longer difference between adjacent over the sequence length: ",sum(max_action_rate)/len(max_action_rate))
    print(len(max_action_diff))
    print("Average sequence lenght: ", int(sum(seq_duration) / len(seq_duration)), " Frames/seq")
    print(len(seq_duration))
    print(f"NUMBER OF frames of all repos (including the unannotated): {duration}, duration: {duration/fps/3600} hours")
    print("Annotation rate (all annotated/all repos): ", (number_of_annotated_images / duration)*fps, " fps")
    from matplotlib import pyplot as plt
    #plt.hist(sorted(fr_per_seq),bins=100)
    #plt.show()

def split_seq_into_2_masks_pairs(folder):
    import os
    import glob
    import cv2
    import numpy as np
    from PIL import Image
    from numpy import asarray
    import shutil
    index = 0
    for subdir, dirs, files in sorted(os.walk(folder)):
        index = 0
        if (subdir != folder):
            path480 = subdir.replace("/masks/", "/masks/")
            print(path480)
            i=0
            seq_files = sorted(glob.glob(path480 + '/*.png'))
            for i in range(0,len(seq_files)-1):
                pairs_folder = os.path.join(os.path.join("/".join(path480.split("/")[:-2]),"pairs_masks"),seq_files[i].split("/")[-2]+"_"+'{:04d}'.format(i))
                os.makedirs(pairs_folder, exist_ok=True, mode=0o777)
                # Source path
                for k in [0,1]:
                    source = seq_files[i+k]
                    destination = os.path.join(pairs_folder,source.split("/")[-1])
                    shutil.copyfile(source, destination)
                start_frame = seq_files[i].split("/")[-1].split(".")[0]
                end_frame = seq_files[i+1].split("/")[-1].split(".")[0]
                print(f"start: {start_frame} and end: {end_frame}")
                image_path480 = path480.replace("/masks/", "/images_48050fps/")
                seq_image_files = sorted(glob.glob(image_path480 + '/*.jpg'))
                #start_index = [i for i, elem in enumerate(seq_image_files) if start_frame.split("_")[-1] in elem][0]
                #end_index = [i for i, elem in enumerate(seq_image_files) if end_frame.split("_")[-1] in elem][0]
                #pairs_folder_image = os.path.join(os.path.join("/".join(image_path480.split("/")[:-2]), "pairs_images"),
                #                            seq_files[i].split("/")[-2] + "_" + str(i))
                #os.makedirs(pairs_folder_image, exist_ok=True, mode=0o777)
                #for im_index in range(start_index,end_index+1):
                #        source = seq_image_files[im_index]
                #        destination = os.path.join(pairs_folder_image, source.split("/")[-1])
                #        shutil.copyfile(source, destination)




def actions_histogram(csv_file, json_folder):
    import pandas as pd
    import json
    import glob
    import collections, functools, operator
    from PIL import Image
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    ##df = pd.read_csv(csv_file)
    df = filter_acions_to_existing_videos(csv_file,json_folder)

    #df = df[df.video_id == vid]  # filter the dataset to just include the video we're targeting
    df = df[["video_id", "start_frame", "stop_frame","narration"]].dropna() #select the important colunms for this task
    df = df.sort_values(by=['stop_frame'])
    df['count'] = 0
    n = 1
    for index, row in df.iterrows():
        start = row["start_frame"]
        stop = row["stop_frame"]
        flag=0
        for infile in sorted(glob.glob(json_folder + '/*.json')):
            f = open(infile, 'r')
            annotation_info = json.load(f)
            vid = "_".join(infile.split("/")[-1].split("_")[1:3])
            #print(vid)
            #v_df = df[df.video_id == vid]
            for key in sorted(annotation_info.keys()):
                frame = int(key.split("/")[-1].split(".")[0].split("_")[-1])
                if(((start)<=int(frame)) & ((stop)>=int(frame)) & (df.at[index,"video_id"] == vid)):
                    #row["count"] = row["count"] + 1
                    df.at[index,"count"] = df.at[index,"count"] + 1
                    flag = 1
                #for i, v in actions.iterrows():
                #    c = df.iloc[i]["count"]
                #    df.iloc[i]["count"] = (c + 1)
                #    print(df.iloc[i])
                #print("")
        #print (df)
        print(f"{round(n / len(df.index) * 100, 1)} %")
        n = n + 1
    df.to_csv("filtered_"+csv_file, index=False)
    #print(df)


def filter_acions_to_existing_videos(csv_file, json_folder):
    import pandas as pd
    import json
    import glob
    import collections, functools, operator
    from PIL import Image
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    df = pd.read_csv(csv_file)

    # df = df[df.video_id == vid]  # filter the dataset to just include the video we're targeting
    #df = df.sort_values(by=['stop_frame'])
    new_df = pd.DataFrame(columns=df.columns)
    all_vids=[]
    for infile in sorted(glob.glob(json_folder + '/*.json')):
        vid = "_".join(infile.split("/")[-1].split("_")[1:3])
        if vid not in all_vids:
            all_vids.append(vid)
            new_df = new_df.append(df[df.video_id == vid])
    #new_df.to_csv("filtered_"+csv_file, index=False)
    return  new_df

def out_of_actions_frames(csv_file,json_folder):
    import pandas as pd
    import glob
    new_df = pd.DataFrame(columns=['vid','frame'])
    df = filter_acions_to_existing_videos(csv_file, json_folder)
    n = 1
    number_of_missing = 0
    number_of_images = 0
    for infile in sorted(glob.glob(json_folder + '/*.json')):
        number_of_missing_per_video = 0
        number_of_images_per_video = 0
        f = open(infile, 'r')
        annotation_info = json.load(f)
        vid = "_".join(infile.split("/")[-1].split("_")[1:3])
        # print(vid)
        v_df = df[df.video_id == vid]
        for key in sorted(annotation_info.keys()):
            frame = int(key.split("/")[-1].split(".")[0].split("_")[-1])
            flag = 0
            min_f = 10000 #big number
            for index, row in v_df.iterrows():
                start = row["start_frame"]
                stop = row["stop_frame"]
                if (((start) <= int(frame)) & ((stop) >= int(frame)) & (df.at[index, "video_id"] == vid)):
                    # row["count"] = row["count"] + 1
                    flag = 1
                    break
                elif (df.at[index, "video_id"] == vid):
                    x = abs(int(frame) - start)
                    y = abs(int(frame) - stop)
                    if min(x,y) < min_f:
                        min_f = min(x,y)
            if flag == 0:

                new_df = new_df.append({'vid': vid, 'frame': frame, 'distance_to_closest':min_f},ignore_index=True)
                number_of_missing = number_of_missing + 1
                number_of_missing_per_video = number_of_missing_per_video + 1
            number_of_images = number_of_images + 1
            number_of_images_per_video = number_of_images_per_video + 1
        print(f"{round(n / len(glob.glob(json_folder + '/*.json')) * 100, 1)} %")
        print(f"{vid} Percentage of out of actions frames ({number_of_missing_per_video} out of {number_of_images_per_video}) = ",
              number_of_missing_per_video / number_of_images_per_video)

        n = n + 1
    new_df.to_csv("out_of_actions_frames_" + csv_file, index=False)
    print(f"Percentage of out of actions frames ({number_of_missing} out of {number_of_images}) = ",number_of_missing/number_of_images)


def generate_videos_from_all_masks_STM(input_directory,output_directory,frame_rate):
    import glob
    import os
    os.makedirs(output_directory,exist_ok=True)
    prev_video = ""
    for video in sorted(glob.glob(input_directory + '/*')):
        if ("_".join(video.split("/")[-1].split("_")[:-3]) != prev_video):
            seq_to_video_modified("/".join(video.split("/")[:-1])+"/"+"_".join(video.split("/")[-1].split("_")[:-3]),output_directory,frame_rate) #video like "'segmentations/0002_P06_101'"
            prev_video = "_".join(video.split("/")[-1].split("_")[:-3])

def actions_to_seq_csv(csv_actions, csv_seq,json_folder):
    ### IMPORTANT: this fucnction is just for 1 part repos
    import pandas as pd
    import json
    import glob
    import collections, functools, operator
    from PIL import Image
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    new_df = pd.DataFrame(columns=['video_id','seq', 'start_frame', 'stop_frame','narration'])
    for infile in sorted(glob.glob(json_folder + '/*.json')):
        df = filter_acions_to_existing_videos(csv_actions, json_folder)
        vid = "_".join(infile.split("/")[-1].split("_")[1:3])
        df2 = df[df.video_id == vid]
        # df = df[df.video_id == vid]  # filter the dataset to just include the video we're targeting
        df2 = df2.sort_values(by=['stop_frame'])
        actions_per_sequence = 3
        i=1
        while not df2.empty:
            seq = df2.head(actions_per_sequence)
            min_value = seq.start_frame.min()
            max_value = seq.stop_frame.max()
            name = vid+'_Part_' + str(1).zfill(3) + '_seq_'+str(i).zfill(3)
            new_df = new_df.append({'video_id':vid,'seq':name,'start_frame':min_value,'stop_frame':max_value,'narration':'-'},ignore_index=True)
            df2 = df2[(df2.start_frame > max_value)]
            i = i +1
        new_df.to_csv(csv_seq, index=False)

def overlap_only(file1, file2): #find the difference between them (NOT WORKING)
    import pandas as pd
    import json
    import glob

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    overlap = pd.concat([df1, df2]).drop_duplicates(keep=False)
    overlap.to_csv("overlap_"+file1, index=False)


def actions_histogram2(csv_file, json_folder):
    import pandas as pd
    import json
    import glob
    import collections, functools, operator
    from PIL import Image
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    ##df = pd.read_csv(csv_file)
    df = filter_acions_to_existing_videos(csv_file,json_folder)

    #df = df[df.video_id == vid]  # filter the dataset to just include the video we're targeting
    df = df[["video_id", "start_frame", "stop_frame","narration"]].dropna() #select the important colunms for this task
    df = df.sort_values(by=['stop_frame'])
    df['count'] = 0
    n = 1
    for index, row in df.iterrows():
        start = row["start_frame"]
        stop = row["stop_frame"]
        flag=0
        for infile in sorted(glob.glob(json_folder + '/*.json')):
            f = open(infile, 'r')
            annotation_info = json.load(f)
            vid = "_".join(infile.split("/")[-1].split("_")[1:3])
            #print(vid)
            #v_df = df[df.video_id == vid]
            for key in sorted(annotation_info.keys()):
                frame = int(key.split("/")[-1].split(".")[0].split("_")[-1])
                if(((start)<=int(frame)) & ((stop)>=int(frame)) & (df.at[index,"video_id"] == vid)):
                    #row["count"] = row["count"] + 1
                    df.at[index,"count"] = df.at[index,"count"] + 1
                    flag = 1
                #for i, v in actions.iterrows():
                #    c = df.iloc[i]["count"]
                #    df.iloc[i]["count"] = (c + 1)
                #    print(df.iloc[i])
                #print("")
        #print (df)
        print(f"{round(n / len(df.index) * 100, 1)} %")
        n = n + 1
    df.to_csv("filtered_"+csv_file, index=False)
    #print(df)


def combine_all_parts_jsons_torento(folder):
    import json
    import glob
    import os
    all_annotations=[]
    prev_file = ''
    for infile in sorted(glob.glob(folder + '/*.json')):
        file2 = infile.split('/')[-1].split('_')[0] # it will return the index number of the repo: e.g. 00001 
        f = open(infile, 'r')
        siize = json.load(f)
        
        #print(prev_file,' ==> ',file)
        if file2 == prev_file and prev_file != '':
            f = open(prev_file_full_name, 'r')
            annotation_info = json.load(f)
            all_annotations.extend(annotation_info)
            f.close()
            print('remove: ', prev_file_full_name)
            os.remove(prev_file_full_name)
        elif prev_file != '' and len(all_annotations) != 0:

            f = open(prev_file_full_name, 'r')
            annotation_info = json.load(f)
            all_annotations.extend(annotation_info)
            f.close()
            print("Number of annotations: ",len(all_annotations), 'saved as ',prev_file_full_name)
            file  = open(prev_file_full_name,'w')
            all = json.dumps(all_annotations)
            file.write(str(all))
            file.close()
            all_annotations=[]
        prev_file = file2
        prev_file_full_name  = infile
        print(infile.split('/')[-1],  ' ',len(siize))

    if len(all_annotations) != 0:
        f = open(prev_file_full_name, 'r')
        annotation_info = json.load(f)
        all_annotations.extend(annotation_info)
        f.close()
        print("Number of annotations: ",len(all_annotations), 'saved as ',prev_file_full_name)
        file  = open(prev_file_full_name,'w')
        all = json.dumps(all_annotations)
        file.write(str(all))
        file.close()

def download_video_series_55(video):
    video_links = ["https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/"+video.split("_")[0]+"/"+video+".MP4"]
    for link in video_links:

        '''iterate through all links in video_links
        and download them one by one'''
        
        # obtain filename by splitting url and getting
        # last string
        file_name = '../videos/'+link.split('/')[-1]

        print( "Downloading file:%s"%file_name)
        
        # create response object
        r = requests.get(link, stream = True)
        if r.status_code != 200:
            print ("From test data")
            r = requests.get(link.replace("train","test"), stream = True)
        
        # download started
        chunk_size = 0
        chunk_count = 0
        with open(file_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size = 1024*1024):
                if chunk:
                    f.write(chunk)
                    chunk_size = chunk_size + len(chunk)
                    chunk_count = chunk_count + 1
                    if chunk_count > 100: #every 100 MBit
                        chunk_count = 0
                        print(str(int(chunk_size/1000000)) + " MBit which is : "+  str(round(chunk_size*100/int(r.headers.get('Content-Length')),1)) + "%")

        print(str(int(chunk_size/1000000)) + " MBit which is : "+  str(round(chunk_size*100/int(r.headers.get('Content-Length')),1)) + "%")
        print( "%s downloaded!\n"%file_name )

    print ("All videos downloaded!")
    return

def download_video_series_100(video):
    video_links = ["https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/"+video.split("_")[0]+"/videos/"+video+".MP4"]
    for link in video_links:

        '''iterate through all links in video_links
        and download them one by one'''
        
        # obtain filename by splitting url and getting
        # last string
        file_name = '../videos/'+link.split('/')[-1]

        print( "Downloading file:%s"%file_name)
        
        # create response object
        r = requests.get(link, stream = True)
        if r.status_code != 200:
            print ("From test data")
            r = requests.get(link.replace("train","test"), stream = True)
        
        # download started
        chunk_size = 0
        chunk_count = 0
        with open(file_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size = 1024*1024):
                if chunk:
                    f.write(chunk)
                    chunk_size = chunk_size + len(chunk)
                    chunk_count = chunk_count + 1
                    if chunk_count > 100: #every 100 MBit
                        chunk_count = 0
                        print(str(int(chunk_size/1000000)) + " MBit which is : "+  str(round(chunk_size*100/int(r.headers.get('Content-Length')),1)) + "%")

        print(str(int(chunk_size/1000000)) + " MBit which is : "+  str(round(chunk_size*100/int(r.headers.get('Content-Length')),1)) + "%")
        print( "%s downloaded!\n"%file_name )

    print ("All videos downloaded!")
    return

def copy_jpg_image(image_name,in_path,out_path):
    image_name = image_name.replace('png','jpg')
    shutil.copy(os.path.join(in_path,image_name),os.path.join(out_path,image_name))


def generate_dense_images_from_masks(input_dir,output_dir):
    import glob
    import os
    #files in this input_dir are like:P30_107/P30_Part_001/P30_107_seq_00001/frame_0000000001/frame_0000000001.jpg
    
    seq_name = ""
    seq_start_frame_str=""
    seq_end_frame_str = ""
    seq_start_frame=0
    seq_end_frame = 0
    video_path = ''
    for infile in sorted(glob.glob(input_dir + '*/*.png')):
        print (infile)
        v = '_'.join(infile.split("/")[-1].split('_')[1:3])
        video_path = '../videos/' + v+'.MP4'
        if not os.path.exists(video_path):
            if len(v.split('_')[1]) == 2:
                download_video_series_55(v)
            else:
                download_video_series_100(v)
        if seq_name != infile.split("/")[-2]: #check if the current seq is not the same with previous one (new seq)
           if seq_name != "" : # if it is not the before starting (seq = "" as inital value)
                #print(len(os.walk("/".join(infile.split("/")[:-2])).__next__()[1]))
                #os.makedirs(os.path.join(output_dir,seq_name+"_"+str(seq_start_frame)+"_"+str(seq_end_frame)), exist_ok=True) this for start and end frame
                os.makedirs(os.path.join(output_dir,seq_name), exist_ok=True)
                print("Seq: ",seq_name)
                print("Start: ",seq_start_frame_str)
                print("Start: ",seq_start_frame)
                print("END: ",seq_end_frame)
                print("-----------------------")
                current_v = '_'.join(seq_start_frame_str.split('_')[1:3])

                generate_images_n_V2(video_path.replace(v,current_v), os.path.join(output_dir,seq_name), seq_start_frame_str,
                                     seq_end_frame - seq_start_frame + 1)
                rename_images(os.path.join(output_dir, seq_name), seq_start_frame_str,seq_start_frame,seq_end_frame - seq_start_frame + 1)

                copy_jpg_image(seq_start_frame_str,"./Images_flat",os.path.join(output_dir,seq_name))
                copy_jpg_image(seq_end_frame_str,"./Images_flat",os.path.join(output_dir,seq_name))
                #generate_images_n_V2_index1(video_path, os.path.join(output_dir,seq_name), seq_start_frame_str,1)
                #rename_images(os.path.join(output_dir, seq_name), seq_start_frame_str,seq_start_frame,1)

                #generate_images_n_V2_index1(video_path, os.path.join(output_dir,seq_name), seq_start_frame_str.replace(str(seq_start_frame),str(seq_end_frame)),1)
                #rename_images(os.path.join(output_dir, seq_name), seq_start_frame_str.replace(str(seq_start_frame),str(seq_end_frame)),seq_end_frame,1)
                
                seq_start_frame_str=infile.split("/")[-1]
                seq_start_frame = int(seq_start_frame_str[-14:-4])
           else: # if it is the first iternation, then assign the start frame
                seq_start_frame_str=infile.split('/')[-1]
                seq_start_frame = int(seq_start_frame_str[-14:-4])

        seq_name = infile.split("/")[-2]  # set the new seq
        seq_end_frame_str = infile.split("/")[-1] #store each frame as a final frame until the above if statement comes true
        seq_end_frame = int(seq_end_frame_str[-14:-4])

    if os.path.exists(video_path):
        #for final folder
        print("Seq: ", seq_name)
        print("Start: ", seq_start_frame_str)
        print("Start: ", seq_start_frame)
        print("END: ", seq_end_frame)
        print("-----------------------")
        os.makedirs(os.path.join(output_dir,seq_name), exist_ok=True)
        generate_images_n_V2(video_path, os.path.join(output_dir,seq_name), seq_start_frame_str,
                             seq_end_frame - seq_start_frame + 1)
        rename_images(os.path.join(output_dir, seq_name), seq_start_frame_str,seq_start_frame,seq_end_frame - seq_start_frame + 1)

        copy_jpg_image(seq_start_frame_str,'./Images_flat',os.path.join(output_dir,seq_name))
        copy_jpg_image(seq_end_frame_str,'./Images_flat',os.path.join(output_dir,seq_name))


def keep_similar_objects(folder):
    import os
    import glob
    from PIL import Image
    import pandas as pd
    import shutil


    num_of_masks_before_common = 0
    num_of_masks_after_common = 0
    t_total = 0
    df = pd.read_csv('data_mapping.csv')
    
    for filename in sorted(os.listdir(folder)):
        f = os.path.join(folder, filename)
        
        # checking if it is a file
        if os.path.isdir(f):

            print(filename)
            images = glob.glob(f+'/*.png')
            vid = (int(images[0].split('/')[-1].split('_')[0]))
            df2 = df[df.video_id == vid]
            u1,u2,common,t= find_common_values(images[0],images[1],df2)            
            num_of_masks_before_common += u1 + u2
            num_of_masks_after_common += common + common
            t_total = t_total+t
            if common == 0: #then delete the interpolation as no shared masks
                shutil.rmtree(f)
                print('deleted: ',filename)
            else:
                with open("repos.txt", "a") as myfile:
                    myfile.write(filename)
                    myfile.write('\n')
                    myfile.close()

            
    print('number of masks before delete the uncommon values: ', num_of_masks_before_common)
    print('number of masks after delete the uncommon values: ', num_of_masks_after_common)
    print('number of deleted masks: ', t_total)
def find_common_values(image1_path, image2_path,df):   
    import numpy as np
    from PIL import Image
    import pandas as pd
    t=0
    image1 = Image.open(image1_path, 'r')
    image2 = Image.open(image2_path, 'r')
    objects_keys = {}
    davis_palette = np.repeat(np.expand_dims(np.arange(0, 256), 1), 3, 1).astype(np.uint8)
    davis_palette[:22, :] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                                 [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                                 [64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],
                                 [64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],
                                 [0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],
                                 [0, 64, 128], [128, 64, 128]]
    x1 = np.array(image1)
    x2 = np.array(image2)
    u1 = np.unique(x1)
    u2 = np.unique(x2)
    common_values = np.intersect1d(u1,u2).tolist()
    #print('u1: ', u1, ' u2: ', u2, ' common: ', common_values)


    for value in u1:
        if value not in common_values:
            object_row = df[df['unique_index'] == value]
            #print(object_row)
            if len(object_row) >  1:
                print("ERROR!! check the csv file")
                exit(0)

            if (object_row["Object_name"] != "left hand").all() & (object_row["Object_name"] != "right hand").all():
                x1 = np.where(x1 == value,0,x1)
                t +=1
            else:
                common_values.append(value)
                #print(common_values)
            #else:
             #   print('HAND DETECTED: ', object_row)

    for value in u2:
        if value not in common_values:
            object_row = df[df['unique_index'] == value]
            if len(object_row) > 1:
                print("ERROR!! check the csv file")
                exit(0)
            if (object_row["Object_name"] != "left hand").all() & (object_row["Object_name"] != "right hand").all():
                x2= np.where(x2 == value,0,x2)
                t +=1
            else:
                common_values.append(value)
                #print(common_values)
            #else:
                #print('HAND DETECTED:', object_row)

    objs = []
    common_values = sorted(common_values)
    for x in range(len(common_values)):
        common_values[x] += 1000 # just to make sure not to overlap between old and new codes
    print(common_values)
    for value in sorted(common_values):
        if value != 1000:
            objects_keys[value-1000] = common_values.index(value)
            object_row = df[df['unique_index'] == value-1000]
            objs.append(object_row.iloc[0]["Object_name"])
        #print(value, '==>', value,common_values.index(value))
        x1 = np.where(x1 == value-1000,common_values.index(value),x1)
        x2 = np.where(x2 == value-1000,common_values.index(value),x2)
    print(np.unique(x1))
    print(np.unique(x2))
    #x1 = np.where(np.nditer(x1) not in common_values,0,x1)
    #x2 = np.where(x2.any(common_values),x2,0)
    '''
    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            if x1[i][j] in common_values:
                x1[i][j] = common_values.index(x1[i][j])
            else:
                x1[i][j] = 0

            if x2[i][j] in common_values:
                x2[i][j] = common_values.index(x2[i][j])
            else:
                x2[i][j] = 0
'''
    #print('New unique:',np.unique(x1))
    #print('New unique_y:',np.unique(x2))

    if objects_keys:
        data = pd.DataFrame(objects_keys.items(), columns=['old_index', 'new_index'])
        data['video_id'] = image1_path.split('/')[-1].split('.')[0].split('_')[0]
        data['interpolation'] = image1_path.split('/')[-2]
        data['Object_name'] = objs
        if not os.path.isfile('int_data_mapping.csv'):
            data.to_csv('int_data_mapping.csv', index=False,header=['old_index', 'new_index','video_id','interpolation','Object_name'])
        else:
            data.to_csv('int_data_mapping.csv',mode='a', header=False,index=False)

    image1 = Image.fromarray(x1)
    image2 = Image.fromarray(x2)
    image1.putpalette(davis_palette)
    image2.putpalette(davis_palette)
    image1.save(image1_path)
    image2.save(image2_path)

    return len(u1)-1,len(u2)-1,min(len(np.unique(x1))-1,len(np.unique(x2))-1), t #delete the 0 value as it is not a mask (reutnr number masks before and after common)

def main_f():
    #first move the image files to "images" then do the below

    #json_to_masks('completed_annotations_stage2/00084_P04_06_Part_001.json','sample3/masks/')
    map(remove_or_exist, ('repos.txt', 'int_data_mapping.csv', 'data_mapping.csv'))

    combine_all_parts_jsons_torento('completed_annotations_stage2/')
    folder_of_jsons_to_masks('completed_annotations_stage2/','sample4/masks/')
    split_seq_into_2_masks_pairs("sample4/masks")  # essentially: copying
    keep_similar_objects('sample4/pairs_masks')
    generate_dense_images_from_masks('sample4/pairs_masks/','sample4/images/')


def remove_or_exist(fname):
    if os.path.exists(fname):
        os.remove(fname)
    

if __name__ == '__main__':
    main_f()