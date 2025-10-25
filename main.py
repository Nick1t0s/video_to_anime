import os
import shutil
import parse_video
import frame_to_anime
import create_video
import sys
import pathlib

args = sys.argv
if(len(args) == 1):
    video_name = input("Enter video file name: ")
    while True:
        if not os.path.isfile(video_name):
            print("File does not exist")
            video_name = input("Enter video file name: ")
        else: break

    temp_path = "temp"
    while os.path.isdir(temp_path):
        print(f"Directory '{temp_path}' exists, the data in it will be deleted.")
        res = input("Do you want to continue? (y/n): ")
        if res == "n":
            temp_path = input("Please specify the directory for temporary files: ")
        else: break

    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    os.mkdir(temp_path)

    anime_temp_path = "results"
    while os.path.isdir(anime_temp_path):
        print(f"Directory '{anime_temp_path}' exists, the data in it will be deleted.")
        res = input("Do you want to continue? (y/n): ")
        if res == "n":
            anime_temp_path = input("Please specify the directory for temporary files: ")
        else: break

    if os.path.exists(anime_temp_path):
        shutil.rmtree(anime_temp_path)
    os.mkdir(anime_temp_path)

    res_name = input("Enter result video file name (press enter to skip): ")
    if res_name == "":
        point_index = video_name.rfind(".")
        res_name = video_name[:point_index]
        file_extension = video_name[point_index:]
        res_name += "_anime"
        res_name += file_extension

    while os.path.isfile(res_name):
        print(f"File '{res_name}' exists, the data in it will be deleted.")
        res = input("Do you want to continue? (y/n): ")
        if res == "n":
            res_name = input("Enter video file name: ")
        else: break

    if os.path.isfile(res_name):
        os.remove(res_name)

    print(video_name)
    print(temp_path)
    print(anime_temp_path)
    print(res_name)


else:
    video_name = args[1]
    if not os.path.isfile(video_name):
        print("File does not exist")
        exit(1)

    temp_path = "temp"
    anime_temp_path = "results"
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    os.mkdir(temp_path)

    res_name = args[2]
    if os.path.isfile(res_name):
        os.remove(res_name)

    print(video_name)
    print(res_name)

if not parse_video.vid_to_frames(video_name, temp_path):
    print("Couldn't extract frames from video")
    exit(1)

if not frame_to_anime.frame_to_anime(temp_path, anime_temp_path):
    print("Couldn't convert frames to anime")
    exit(1)

if not create_video.assemble(video_name, anime_temp_path, res_name):
    print("Couldn't assemble video")
    exit(1)

print("Done")