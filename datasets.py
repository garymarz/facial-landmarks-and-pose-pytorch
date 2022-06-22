import numpy as np
import cv2
import sys
import torch
import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from PIL import Image, ImageFilter
import utils

sys.path.append('..')

from torch.utils import data
from torch.utils.data import DataLoader


def flip(img, annotation):
    img = np.fliplr(img).copy()
    h, w = img.shape[:2]

    x_min, y_min, x_max, y_max = annotation[0:4]
    landmark_x = annotation[4::2]
    landmark_y = annotation[4 + 1::2]

    bbox = np.array([w - x_max, y_min, w - x_min, y_max])
    for i in range(len(landmark_x)):
        landmark_x[i] = w - landmark_x[i]

    new_annotation = list()
    new_annotation.append(x_min)
    new_annotation.append(y_min)
    new_annotation.append(x_max)
    new_annotation.append(y_max)

    for i in range(len(landmark_x)):
        new_annotation.append(landmark_x[i])
        new_annotation.append(landmark_y[i])

    return img, new_annotation


def channel_shuffle(img, annotation):
    if (img.shape[2] == 3):
        ch_arr = [0, 1, 2]
        np.random.shuffle(ch_arr)
        img = img[..., ch_arr]
    return img, annotation


def random_noise(img, annotation, limit=[0, 0.2], p=0.5):
    if random.random() < p:
        H, W = img.shape[:2]
        noise = np.random.uniform(limit[0], limit[1], size=(H, W)) * 255

        img = img + noise[:, :, np.newaxis] * np.array([1, 1, 1])
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img, annotation


def random_brightness(img, annotation, brightness=0.3):
    alpha = 1 + np.random.uniform(-brightness, brightness)
    img = alpha * image
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, annotation


def random_contrast(img, annotation, contrast=0.3):
    coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
    alpha = 1.0 + np.random.uniform(-contrast, contrast)
    gray = img * coef
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    img = alpha * img + gray
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, annotation


def random_saturation(img, annotation, saturation=0.5):
    coef = nd.array([[[0.299, 0.587, 0.114]]])
    alpha = np.random.uniform(-saturation, saturation)
    gray = img * coef
    gray = np.sum(gray, axis=2, keepdims=True)
    img = alpha * img + (1.0 - alpha) * gray
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, annotation


def random_hue(image, annotation, hue=0.5):
    h = int(np.random.uniform(-hue, hue) * 180)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 180
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image, annotation


def scale(img, annotation):
    f_xy = np.random.uniform(-0.4, 0.8)
    origin_h, origin_w = img.shape[:2]

    bbox = annotation[0:4]
    landmark_x = annotation[4::2]
    landmark_y = annotation[4 + 1::2]

    h, w = int(origin_h * f_xy), int(origin_w * f_xy)
    image = resize(img, (h, w),
                   preserve_range=True,
                   anti_aliasing=True,
                   mode='constant').astype(np.uint8)

    new_annotation = list()
    for i in range(len(bbox)):
        bbox[i] = bbox[i] * f_xy
        new_annotation.append(bbox[i])

    for i in range(len(landmark_x)):
        landmark_x[i] = landmark_x[i] * f_xy
        landmark_y[i] = landmark_y[i] * f_xy
        new_annotation.append(landmark_x[i])
        new_annotation.append(landmark_y[i])

    return image, new_annotation


def rotate(img, annotation, alpha=30):

    bbox = annotation[0:4]
    landmark_x = annotation[4::2]
    landmark_y = annotation[4 + 1::2]
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat,
                                          (img.shape[1], img.shape[0]))

    point_x = [bbox[0], bbox[2], bbox[0], bbox[2]]
    point_y = [bbox[1], bbox[3], bbox[3], bbox[1]]

    new_point_x = list()
    new_point_y = list()
    for (x, y) in zip(landmark_x, landmark_y):
        new_point_x.append(rot_mat[0][0] * x + rot_mat[0][1] * y +
                           rot_mat[0][2])
        new_point_y.append(rot_mat[1][0] * x + rot_mat[1][1] * y +
                           rot_mat[1][2])

    new_annotation = list()
    new_annotation.append(min(new_point_x))
    new_annotation.append(min(new_point_y))
    new_annotation.append(max(new_point_x))
    new_annotation.append(max(new_point_y))

    for (x, y) in zip(landmark_x, landmark_y):
        new_annotation.append(rot_mat[0][0] * x + rot_mat[0][1] * y +
                              rot_mat[0][2])
        new_annotation.append(rot_mat[1][0] * x + rot_mat[1][1] * y +
                              rot_mat[1][2])

    return img_rotated_by_alpha, new_annotation

def get_R(x,y,z):
    ''' Get rotation matrix from three rotation angles (radians). right-handed.
    Args:
        angles: [3,]. x, y, z angles
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    # x
    
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    # y
    
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    # z
    
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R

def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    print(file_path)
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

class WLFWDatasets(data.Dataset):
    def __init__(self, file_list, transforms=None):
        self.line = None
        self.path = None
        self.landmarks = None
        self.attribute = None
        self.filenames = None
        self.euler_angle = None
        self.transforms = transforms
        with open(file_list, 'r') as f:
            self.lines = f.readlines()

    def __getitem__(self, index):
        self.line = self.lines[index].strip().split()
        self.img = Image.open(self.line[0]).convert('RGB')
        self.landmark = np.asarray(self.line[1:197], dtype=np.float32)
        self.attribute = np.asarray(self.line[197:203], dtype=np.int32)
        self.euler_angle = [float(i)/180*np.pi for i in self.line[203:206]]
       
        self.R = get_R(self.euler_angle[0],self.euler_angle[1],self.euler_angle[2])
        self.euler_angle_labels = torch.FloatTensor([self.euler_angle[0],self.euler_angle[1],self.euler_angle[2]]) # y,p, r
        if self.transforms:
            self.img = self.transforms(self.img)
        return (self.img, self.landmark, self.attribute, torch.FloatTensor(self.R))

    def __len__(self):
        return len(self.lines)

class test_WLFWDatasets(data.Dataset):
    def __init__(self, file_list, transforms=None):
        self.line = None
        self.path = None
        self.landmarks = None
        self.attribute = None
        self.filenames = None
        self.euler_angle = None
        self.transforms = transforms
        with open(file_list, 'r') as f:
            self.lines = f.readlines()

    def __getitem__(self, index):
        self.line = self.lines[index].strip().split()
        self.img = Image.open(self.line[0]).convert('RGB')
        self.landmark = np.asarray(self.line[1:197], dtype=np.float32)
        self.attribute = np.asarray(self.line[197:203], dtype=np.int32)
        self.euler_angle = [float(i)/180*np.pi for i in self.line[203:206]]
       
        self.R = get_R(self.euler_angle[0],self.euler_angle[1],self.euler_angle[2])
        self.euler_angle_labels = torch.FloatTensor([self.euler_angle[0],self.euler_angle[1],self.euler_angle[2]]) # y,p, r
        if self.transforms:
            self.img = self.transforms(self.img)
        return (self.img, self.landmark, self.attribute, torch.FloatTensor(self.R), self.euler_angle_labels)

    def __len__(self):
        return len(self.lines)
    
# if __name__ == '__main__':
#     file_list = './data/test_data/list.txt'
#     wlfwdataset = WLFWDatasets(file_list)
#     dataloader = DataLoader(wlfwdataset,
#                             batch_size=256,
#                             shuffle=True,
#                             num_workers=0,
#                             drop_last=False)
#     for img, landmark, attribute, euler_angle in dataloader:
#         print("img shape", img.shape)
#         print("landmark size", landmark.size())
#         print("attrbute size", attribute)
#         print("euler_angle", euler_angle.size())
      
class Pose_300W_LP(Dataset):
    # Head pose from 300W-LP dataset
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])

        # k = 0.2 to 0.40
        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi

        # Flip?
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Blur?
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)

        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

        # Get target tensors
        labels = binned_pose
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]
    
    def __len__(self):
        # 122,450
        return self.length
         
        

   
    
class Pose_300W_LP0(Dataset):
    # Head pose from 300W-LP dataset
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(
            self.data_dir, self.X_train[index] + self.img_ext)).convert('RGB')
        img = img.convert(self.image_mode)
        mat_path = os.path.join(
            self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0, :])
        y_min = min(pt2d[1, :])
        x_max = max(pt2d[0, :])
        y_max = max(pt2d[1, :])

        # k = 0.2 to 0.40
        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] # * 180 / np.pi
        yaw = pose[1] #* 180 / np.pi
        roll = pose[2] # * 180 / np.pi
        
        # Gray images

        # Flip?
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Blur?
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)

        # Add gaussian noise to label
        #mu, sigma = 0, 0.01 
        #noise = np.random.normal(mu, sigma, [3,3])
        #print(noise) 

        # Get target tensors
        R = utils.get_R(pitch, yaw, roll)#+ noise
        labels = torch.FloatTensor([yaw, pitch, roll])
        #labels = torch.FloatTensor([temp_l_vec, temp_b_vec, temp_f_vec])

        if self.transform is not None:
            img = self.transform(img)

        return img, [], [], torch.FloatTensor(R), labels, self.X_train[index]

    def __len__(self):
        # 122,450
        return self.length
    
    
    
    
# class AFLW2000(Dataset):
#     def __init__(self, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
#         self.data_dir = 'data/AFLW2000/'
#         self.transform = transform
#         self.img_ext = img_ext
#         self.annot_ext = annot_ext
#         filename_list = get_list_from_filenames('data/AFLW2000/files.txt')

#         self.X_train = filename_list
#         self.y_train = filename_list
#         self.image_mode = image_mode
#         self.length = len(filename_list)
        
#     def __getitem__(self, index):
        
#         img = Image.open(self.data_dir+self.X_train[index] + self.img_ext)
#         img = img.convert(self.image_mode)
#         mat_path = 'data/AFLW2000'+ self.y_train[index] + self.annot_ext

#         # Crop the face loosely
#         pt2d = utils.get_pt2d_from_mat(mat_path)

#         x_min = min(pt2d[0,:])
#         y_min = min(pt2d[1,:])
#         x_max = max(pt2d[0,:])
#         y_max = max(pt2d[1,:])

#         k = 0.20
#         x_min -= 2 * k * abs(x_max - x_min)
#         y_min -= 2 * k * abs(y_max - y_min)
#         x_max += 2 * k * abs(x_max - x_min)
#         y_max += 0.6 * k * abs(y_max - y_min)
#         img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

#         # We get the pose in radians
#         pose = utils.get_ypr_from_mat(mat_path)
#         # And convert to degrees.
#         pitch = pose[0]# * 180 / np.pi
#         yaw = pose[1] #* 180 / np.pi
#         roll = pose[2]# * 180 / np.pi
     
#         R = utils.get_R(pitch, yaw, roll)

#         labels = torch.FloatTensor([yaw, pitch, roll])


#         if self.transform is not None:
#             img = self.transform(img)

#         return img, torch.FloatTensor(R), labels, self.X_train[index]

#     def __len__(self):
#         # 2,000
#         return self.length
    
    
class AFLW2000(Dataset):
    def __init__(self,  filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = 'data/AFLW2000/'
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames('data/AFLW2000/files.txt')

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(self.data_dir+self.X_train[index] + self.img_ext)
        img = img.convert(self.image_mode)
        mat_path = 'data/AFLW2000'+ self.y_train[index] + self.annot_ext

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)

        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])

        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # 2,000
        return self.length