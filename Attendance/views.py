from django import template
from django.shortcuts import render
import cv2
import os
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from .models import Patient


#Render home Page#
def home(request):
    return render(request, 'home.html')


def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[0]
    current_time = datetime.now().strftime("%H:%M:%S")

    # create new patient instance
    patient = Patient(first_name='', last_name=username , date_of_birth=datetime.now().date(),
                      insurance_info='', access_id=userid)

    # save patient to database
    patient.save()

    # log attendance in CSV file
    df = pd.read_csv(f'Rollcall/Attendance-{datetoday}.csv')
    if userid not in list(df['ID']):
        with open(f'Rollcall/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')



# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier(
    'static/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# If these directories don't exist, create them

# create directories if they don't exist
if not os.path.isdir('Rollcall'):
    os.makedirs('Rollcall')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

# create file if it doesn't exist
attendance_file = f'Rollcall/Attendance-{datetoday}.csv'
if attendance_file not in os.listdir('Rollcall'):
    with open(attendance_file, 'w') as f:
        f.write('Name,ID,Time')


# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


# extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points


# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Rollcall/Attendance-{datetoday}.csv')
    names = df['Name']
    ID = df['ID']
    times = df['Time']
    l = len(df)
    return names,ID,times,l

# Add Attendance of a specific user



# This function will run when we click on Take Attendance Button
def start(request):
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render(request, 'home.html', totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret, frame = cap.read()
        if extract_faces(frame) != ():
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, id , times, l = extract_attendance()
    context = {
        'nameS': names,
        'ID': id,
        'times': times,
        'l': l,
        'totalreg': totalreg(),
        'datetoday2': datetoday2
    }
    return render(request, 'home.html', context)


# This function will run when we add a new user

@csrf_exempt
def add(request):
    if request.method == 'POST':
        Last_name = str(request.POST.get('last_name'))
        Access_id = str(request.POST.get('access_id'))
        userimagefolder = 'static/faces/'+str(Last_name)+'_'+Access_id
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        cap = cv2.VideoCapture(0)
        i, j = 0, 0
        while 1:
            _, frame = cap.read()
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/50', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 10 == 0:
                    name = Last_name+'_'+Access_id+'.jpg'
                    cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                    i += 1
                j += 1
            if j == 500:
                break
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        print('Training Model')
        train_model()
        names, rolls, times, l = extract_attendance()
        return render(request, 'home.html', {'names': names, 'rolls': rolls, 'times': times, 'l': l, 'totalreg': totalreg(), 'datetoday2': datetoday2})
    else:
        return HttpResponse('Invalid Request Method')



