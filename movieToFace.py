# こんにゃくさんの記事をとても参考にしています。
# https://konnyaku256.com/tweepy/newgame

import cv2
import glob
 
video_path = './prichan10.mov'
video_name = video_path[2:9] + '_'
output_path = './captures/'
out_face_path = './faces/'
xml_path = "./lbpcascade_animeface.xml"
 
def convert_movie_to_image(num_cut):
 
    capture = cv2.VideoCapture(video_path)
 
    image_count = 0
    frame_count = 0
 
    while(capture.isOpened()):
 
        ret, frame = capture.read()
        if ret == False:
            break
 
        if frame_count % num_cut == 0:
            image_file_name = output_path + str(image_count) + ".jpg"
            cv2.imwrite(image_file_name, frame)
            image_count += 1
 
        frame_count += 1
 
    capture.release()
 
def detect_face(image_list):
 
    classifier = cv2.CascadeClassifier(xml_path)
 
    image_count = 1
    for image_path in image_list:
 
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
 
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
 
        face_points = classifier.detectMultiScale(gray_image, \
                scaleFactor=1.2, minNeighbors=2, minSize=(1,1))
 
        for points in face_points:
 
            x, y, width, height =  points
 
            dist_image = original_image[y:y+height, x:x+width]
 
            face_image = cv2.resize(dist_image, (64,64))
            new_image_name = out_face_path + video_name + str(image_count) + 'face.jpg'
            cv2.imwrite(new_image_name, face_image)
            image_count += 1
 
if __name__ == '__main__':
 
    convert_movie_to_image(int(50))
 
    images = glob.glob(output_path + '*.jpg')
    detect_face(images)
