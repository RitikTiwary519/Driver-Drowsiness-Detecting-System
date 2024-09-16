import cv2
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance

def eye_aspect_ratio(eye):
    A=distance.euclidean(eye[1],eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear=(A+B)/(2.0*C)
    return ear

(lStart,lEnd) = face_utils.FACIAL_LANDMARK_68_IDXA('left_eye')  #using function of landmarks for left eye
(rStart,rEnd) = face_utils.FACIAL_LANDMARK_68_IDXA('right_eye')
#this is ear can be calculated from this function.
#ear remains constant until our eyes are open and drops dratically when we close our eyes
#this can be used easily in face detection among various landmarks form the dlib (ie 64 to be precise)

detect=dlib.get_frontal_face_detector() # fast and active already installed
predict = dlib.shape_predictor("shape_predictor_68_landmarks.dat") #uses landmarks of the face , eyes uses 0,1,2,4,5 and to
# this is will be used to calculate ear of the eye
cap = cv2.VideoCaputre(0) # specifying the primary camera
while True:
    ret,frame=cap.read() #returns boolean value if able to read or not then returns the array of imiginary vector
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #frames to gray scale
    detect(gray,0)
    subjects = detect(gray,0)
    for subject in subjects:
        shape=predict(gray,subjects)
        shape= face_utils.shape_to_np(shape)
        leftEye= shape(lStart:lEnd)
        leftEye = shape[rStart:End]
    cv2.imshow("Frame",frame)
    cv2.waitKey(1)