import argparse

import numpy as np
import cv2
from PIL import Image

import torch
import torchvision

from torchvision import transforms
from face_detection.face_detection import RetinaFace
from models.sd import PFLDInference, AuxiliaryNet
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


checkpoint = torch.load('checkpoint/snapshot/checkpoint_epoch.pth.tar', map_location=device)
pfld_backbone = PFLDInference(backbone_file='',deploy=True,pretrained=False).to(device)
pose_net = AuxiliaryNet().to(device)
pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
pose_net.load_state_dict(checkpoint['auxiliarynet'])
pfld_backbone.eval()
pose_net.eval()
transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


detector = RetinaFace(gpu_id=0)
idx_tensor = [idx for idx in range(66)]
idx_tensor = torch.FloatTensor(idx_tensor).to(device)


cap = cv2.VideoCapture(0)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")
    
    
with torch.no_grad():  
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            break

        height, width = frame.shape[:2]
        #print(frame.shape[:2])
        faces = detector(frame)
        
        for box, _ , score in faces:

            if score < 0.95:
                continue

            x1, y1, x2, y2 = (box[:4] + 0.5).astype(np.int32)

            w = x2 - x1 + 1
            h = y2 - y1 + 1
            cx = x1 + w // 2
            cy = y1 + h // 2

            size = int(max([w, h]) * 1.1)
            x1 = cx - size // 2
            x2 = x1 + size
            y1 = cy - size // 2
            y2 = y1 + size

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            edx1 = max(0, -x1)
            edy1 = max(0, -y1)
            edx2 = max(0, x2 - width)
            edy2 = max(0, y2 - height)

            cropped = frame[y1:y2, x1:x2]
            if (edx1 > 0 or edy1 > 0 or edx2 > 0 or edy2 > 0):
                cropped = cv2.copyMakeBorder(cropped, edy1, edy2, edx1, edx2,
                                             cv2.BORDER_CONSTANT, 0)

            cropped = Image.fromarray(cropped)
            cropped = cropped.convert('RGB')
            cropped = transformations(cropped)

            cropped = torch.Tensor(cropped[None, :]).to(device)
            
            C , R_pred = pfld_backbone(cropped)
            euler = utils.compute_euler_angles_from_rotation_matrices(
                    R_pred)*180/np.pi
            pitch_predicted = euler[:, 0].cpu()
            yaw_predicted = euler[:, 1].cpu()
            roll_predicted = euler[:, 2].cpu()
  
            utils.plot_pose_cube(frame,  yaw_predicted, pitch_predicted, roll_predicted, x1 + int(.5*(
                        x2-x1)), y1 + int(.5*(y2-y1)), size=w)
            
            landmarks = pose_net(C)
            pre_landmark = landmarks[0]
            pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [size, size] - [edx1, edy1]

            for (x, y) in pre_landmark.astype(np.int32):
                cv2.circle(frame, (x1 + x, y1 + y), 1, (0, 0, 255))
        text = 'Yaw:{},Pitch:{},Roll:{}'        
        cv2.putText()
        cv2.imshow('face_landmark', frame)
        if cv2.waitKey(10) == 27:
            break
