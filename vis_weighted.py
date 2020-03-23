# @Author: ajisetyoko <simslab-cs>
# @Date:   2020-01-13T02:19:05+08:00
# @Email:  aji.setyoko.second@gmail.com
# @Last modified by:   simslab-cs
# @Last modified time: 2020-03-03T10:33:15+08:00


import cv2
import numpy as np
import os
import glob
import pickle

train_dir = '/home/simslab-cs/Documents/Dataset/sbu_processed/train/'
test_dir  = '/home/simslab-cs/Documents/Dataset/sbu_processed/test/'

folder_train = (os.listdir(train_dir))
folder_test = (os.listdir(test_dir))

neighbor_link = [(0,1),(1,2),(1,3),(3,4),(4,5),(8,7),(7,6),
                (6,1),(14,13),(13,12),(12,2),(11,10),
                (10,9),(9,2),(9,3),(12,6)]

def draw(x,y,canvas,idx=0):
    assert x.shape[0] == y.shape[0]
    x = x.astype(int)
    y = y.astype(int)

    clr = (255,0,0)
    if idx == 2:
        clr = (0,255,0)

    for point in range(0,x.shape[0]):
        cv2.circle(canvas,(x[point],y[point]), 3, (0,0,255), -1)

    lineThick=2
    for lc in neighbor_link:
        cv2.line(canvas,(x[lc[0]],y[lc[0]]),(x[lc[1]],y[lc[1]]),clr,lineThick)
    return canvas

def per_video_xyz(data):
    data_clean = data[:,1:]
    per_video  = []
    for frame in range (data_clean.shape[0]):
        example = data_clean[frame:frame+1,:]
        # print(example.shape)
        ex      = []
        for person in range(1):
            ex_p = []
            for point in range(15):
                x = point*3+0
                y = point*3+1
                z = point*3+2
                x = example[0,x]
                y = example[0,y]
                z = example[0,z]
                ex_p.append([x,y,z])
            ex.append(ex_p)
            ex_p = []
            for point in range(15,30):
                x = point*3+0
                y = point*3+1
                z = point*3+2
                x = example[0,x]
                y = example[0,y]
                z = example[0,z]
                ex_p.append([x,y,z])
            ex.append(ex_p)
            # ex.append(ex_p)
        per_video.append(ex)
    per_video = np.array(per_video)
    per_video = np.transpose(per_video,(3,0,2,1))
    return per_video

list_file = []
target_list = []
for i in range(len(folder_train)):
    inside_folder = train_dir+folder_train[i]+'/'+folder_train[i]
    each_part = (os.listdir(inside_folder))
    for ii in range(1,len(each_part)):
        file = inside_folder+'/'+each_part[ii]+'/'+'001'
        class_name = each_part[ii]
        file_txt = glob.glob(file + "/*.txt")
        assert len(file_txt)==1
        list_file.append(file_txt)
        target_list.append(class_name)
assert len(list_file) == len(target_list)

test_list_file = []
test_target_list = []
for i in range(len(folder_test)):
    inside_folder = test_dir+folder_test[i]+'/'+folder_test[i]
    each_part = (os.listdir(inside_folder))
    for ii in range(1,len(each_part)):
        file = inside_folder+'/'+each_part[ii]+'/'+'001'
        class_name = each_part[ii]
        file_txt = glob.glob(file + "/*.txt")
        assert len(file_txt)==1
        test_list_file.append(file_txt)
        test_target_list.append(class_name)
assert len(list_file) == len(target_list)

fp = np.zeros((len(list_file),3,26,15,2),dtype=np.float32)
for i,file_read in enumerate(list_file):
    data = np.loadtxt(file_read[0],delimiter=',')
    pervideo = per_video_xyz(data)
    maxi = pervideo.shape[1]
    if maxi>26:
        maxi = 26
        pervideo = pervideo[:,0:maxi,:,:]
    elif maxi<=26:
        for m in range(maxi-1,26-1):
            pervideo = np.append(pervideo,pervideo[:,maxi-1:maxi,:,:],axis=1)
    fp[i,:,0:26,:,:] = pervideo

def vis_coor(coor):
    w,h  = 1280,960
    canvas = np.ones((h,w,3),np.uint8)*255
    p1 = coor[:,:,0]
    x = p1[0,:]
    y = p1[1,:]
    norm_x = (x * 1280)
    norm_y = (y * 960)
    canvas = draw(norm_x,norm_y,canvas)
    p2 = coor[:,:,1]
    x = p2[0,:]
    y = p2[1,:]
    norm_x = (x * 1280)
    norm_y = (y * 960)
    canvas = draw(norm_x,norm_y,canvas)
    cv2.imshow('Windows',canvas)
    k = cv2.waitKey(0)
    if k == 27:
        exit()
    elif k == 32:
        cv2.destroyAllWindows()
        return True
    elif k ==101:
        print(x)
        cv2.destroyAllWindows()
        return True

def avera(c1,c2):
    w,h  = 1280,960
    p1 = c1[:,:,0]
    x = p1[0,:]
    y = p1[1,:]
    norm_xp11 = (x * 1280)
    norm_yp11 = (y * 960)
    p2 = c1[:,:,1]
    x = p2[0,:]
    y = p2[1,:]
    norm_xp21 = (x * 1280)
    norm_yp21 = (y * 960)

    p1 = c2[:,:,0]
    x = p1[0,:]
    y = p1[1,:]
    norm_xp12 = (x * 1280)
    norm_yp12 = (y * 960)
    p2 = c2[:,:,1]
    x = p2[0,:]
    y = p2[1,:]
    norm_xp22 = (x * 1280)
    norm_yp22 = (y * 960)

    nxp1 = (norm_xp11 + norm_xp12 )/2
    nyp1 = (norm_yp11+norm_yp12)/2
    nxp2 = (norm_xp21+norm_xp22)/2
    nyp2 = (norm_yp21+norm_yp22)/2

    canvas = np.ones((h,w,3),np.uint8)*255
    canvas = draw(nxp1,nyp1,canvas,idx=2)
    canvas = draw(nxp2,nyp2,canvas,idx=2)
    cv2.imshow('Windows',canvas)
    k = cv2.waitKey(0)
    if k == 27:
        exit()
    elif k == 32:
        cv2.destroyAllWindows()
        return True
    elif k ==101:
        print(x)
        cv2.destroyAllWindows()
        return True

print(fp[0,:,:,:,:].shape)
video1 = fp[0,:,:,:,:]
new_combine = video1[:,0:1,:,:]
for i in range(video1.shape[1]-1):
    coor1 = video1[:,i  ,:,:]
    coor2 = video1[:,i+1,:,:]
    new_c = (coor1+coor2)/2

    new_c = np.expand_dims(new_c,axis=1)
    coor2 = np.expand_dims(coor2,axis=1)
    new_combine = np.concatenate((new_combine,new_c,coor2),axis=1)
print(new_combine.shape)
video1 = new_combine
new_combine = video1[:,0:1,:,:]
for i in range(video1.shape[1]-1):
    coor1 = video1[:,i  ,:,:]
    coor2 = video1[:,i+1,:,:]
    new_c = (coor1+coor2)/2

    new_c = np.expand_dims(new_c,axis=1)
    coor2 = np.expand_dims(coor2,axis=1)
    new_combine = np.concatenate((new_combine,new_c,coor2),axis=1)
print(new_combine.shape)


for a in range(new_combine.shape[1]):
    coordinate1 = new_combine[:,a,:,:]
    coordinate2 = new_combine[:,a+1,:,:]
    vis_coor(coordinate1)
    new_coordinate = avera(coordinate1,coordinate2)
