from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import os


_face_model : str = 'resource/face_model.pt'

def train_face_recoganize():
    
    #Initialising MTCNN and InceptionResnetV1
    mtcnnfalse=MTCNN(image_size=240,margin=0,keep_all=False,min_face_size=40)
    mtcnn=MTCNN(image_size=240,margin=0,keep_all=True,min_face_size=40)
    resnet=InceptionResnetV1(pretrained='vggface2').eval()

    dataset = datasets.ImageFolder('data/Photos') # photos folder path 
    idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names

    def collate_fn(x):
        return x[0]

    loader = DataLoader(dataset, collate_fn=collate_fn)

    name_list = [] # list of names corrospoing to cropped photos
    embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

    for img, idx in loader:
        face, prob = mtcnnfalse(img, return_prob=True) 
        if face is not None and prob>0.92:
            emb = resnet(face.unsqueeze(0)) 
            embedding_list.append(emb.detach()) 
            name_list.append(idx_to_class[idx])        

    # save data
    data = [embedding_list, name_list] 

    torch.save(data, _face_model) # saving data.pt file

    load_data = data
    # load_data = torch.load(_face_model)
    embedding_list = load_data[0]
    name_list = load_data[1]

    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab Frame! Please try again")
            break

        img = Image.fromarray(frame)
        img_cropped_list, prob_list = mtcnn(img, return_prob=True)
        if img_cropped_list is not None:
            boxes, _ = mtcnn.detect(img)
            for i, prob in enumerate(prob_list):
                if prob > 0.90:
                    emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()

                    dist_list = []  # List of matched distances, minimum distance is used to identify the person

                    for idx, emb_db in enumerate(embedding_list):
                        dist = torch.dist(emb, emb_db).item()
                        dist_list.append(dist)

                    min_dist = min(dist_list)  # Get minimum dist value
                    min_dist_idx = dist_list.index(min_dist)  # Get minimum dist index
                    name = name_list[min_dist_idx]  # Get name corresponding to minimum dist

                    box = boxes[i]

                    original_frame = frame.copy()  # Storing a copy of the frame before drawing on it

                    if min_dist < 0.90:
                        frame = cv2.putText(frame, name + ' ' + str(min_dist), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 0), 1, cv2.LINE_AA)

                    frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        cv2.imshow("Face Detection", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC
            print('Esc pressed, closing...')
            break

        elif k % 256 == 32:  # Space to save image
            print('Enter your name:')
            name = input()

            # Create a directory if not exists
            if not os.path.exists('data/Photos/' + name):
                os.mkdir('data/Photos/' + name)

            img_name = "data/Photos/{}/{}.jpg".format(name, int(time.time()))
            cv2.imwrite(img_name, original_frame)
            print("Saved: {}".format(img_name))

    cam.release()
    cv2.destroyAllWindows()


def train_cat_log_recoganize():
    pass    


if __name__ == "__main__":
    train_face_recoganize()