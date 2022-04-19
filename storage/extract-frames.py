import cv2
import os
import dlib
from imutils import face_utils
import matplotlib.pyplot as plt 
import random
import string

def random_str():
    return ''.join(random.choice(string.ascii_letters) for x in range(12))

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
FRAMES_PER_IMAGE = 300

# https://towardsdatascience.com/facial-mapping-landmarks-with-dlib-python-160abcf7d672
# Uses pretrained model (shape_predictor_68_face_landmarks.dat)
def extract_landmarks(video_path, classification):
    cap = cv2.VideoCapture(video_path)
    count = 0
    plt.axis('off')
    while True:
        _, image = cap.read()
        # Converting the image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Get faces into webcam's image
        rects = detector(gray, 0)
        
        # For each detected face, find the landmark.
        for (i, rect) in enumerate(rects):
            # Make the prediction and transfom it to numpy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
        
            # Draw on our image, all the finded cordinate points (x,y) 
            for (x, y) in shape:
                print(x, y)
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
        
        plt.scatter(shape[:,0], -shape[:,1])
        #plt.show()
        print(classification)
        plt.savefig("./assets/output/{}/{}".format(classification, random_str()))
        plt.clf() 
        count += 1
        if(count > FRAMES_PER_IMAGE): break
        # Show the image
        # cv2.imshow("Output", image)
        
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    
def extract_frames(video_path, output):
    print(video_path, output)
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("./assets/images/{}/%s.jpg".format(output) % random_str(), image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Reading new frame: ', output, video_path)
        if(count > FRAMES_PER_IMAGE): break
        count += 1

#gets all files in a folder
def get_files(path):
    #we shall store all the file names in this list
    filelist = []

    for root, dirs, files in os.walk(path):
        for file in files:
            file_type = file.split(".")
            #append the file name to the list
            if(file_type[len(file_type) - 1] == "mov" or file_type[len(file_type) - 1] == "MOV" or file_type[len(file_type) - 1] == "mp4"):
                filelist.append(os.path.join(root,file))

    return filelist

#gets classfication based on naming
def get_classification(str_path):
    file = str_path.split("/")
    file = file[len(file) - 1].split(".")
    return int(file[0])

def main():
    files = get_files("./assets/videos")

    for video_path in files:
        #print(video_path)
        classification = get_classification(video_path)
        #extract_frames(video_path, classification)
        extract_landmarks(video_path, classification)

if __name__ == "__main__":
    main()