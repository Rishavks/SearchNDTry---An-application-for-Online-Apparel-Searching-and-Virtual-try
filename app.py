# imports for firebase
from flask import Flask, flash, redirect, render_template, request, session, abort, url_for
import pyrebase

# imports for vision api
from flask import Flask, request, render_template, Response
import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path
import os, io
import glob
import cv2

from google.cloud import vision

# imports for virtual try
from tensorflow.keras.models import load_model
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import pixellib
from pixellib.tune_bg import alter_bg
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from Model import Model
from PIL import Image
import time
from matplotlib.image import imread
import urllib.request

# tf.compat.v1.enable_eager_execution()

# google cloud service key
credential_path = "C:/Users/Rishav/OneDrive/Desktop/Minor Project-1/SearchNDTry/servicekey.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

# init client for vision api
client = vision.ImageAnnotatorClient()

# # loading model for virtual try
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# saved = load_model("save_ckp_frozen.h5")

# class fashion_tools(object):
#     def __init__(self,imageid,model,version=1.1):
#         self.imageid = imageid
#         self.model   = model
#         self.version = version
        
#     def get_dress(self,stack=False):
#         """limited to top wear and full body dresses (wild and studio working)"""
#         """takes input rgb----> return PNG"""
#         name =  self.imageid
#         file = cv2.imread(name)
#         # file = tf.image.resize_with_pad(file,target_height=512,target_width=512)
#         # if tf.executing_eagerly():
#         #     rgb = file.numpy()
#         # else:
#         #     rgb = file.eval()
#         # tf.config.run_functions_eagerly(True)
#         file = cv2.resize(file, (512,512))
#         # print(type(file))
#         rgb  = file   
#         #rgb  = file.numpy()
#         file = np.expand_dims(file,axis=0)/ 255.
#         seq = self.model.predict(file)
#         seq = seq[3][0,:,:,0]
#         seq = np.expand_dims(seq,axis=-1)
#         c1x = rgb*seq
#         return c1x
        
        
#     def get_patch(self):
#         return None

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# model = Model("checkpoints/jpp.pb",
#               "checkpoints/gmm.pth",
#               "checkpoints/tom.pth",
#               use_cuda=False)


# Initialze flask constructor
app = Flask(__name__)

# Details for firebase auth
config = {
    "apiKey": "AIzaSyBi2aFx-LhY9wwuJ9DnVJ9swDn75wnQY9E",
    "authDomain": "searchndtry-f0f18.firebaseapp.com",
    #"projectId": "searchndtry-f0f18",
    "storageBucket": "searchndtry-f0f18.appspot.com",
    #"messagingSenderId": "183824581518",
    #"appId": "1:183824581518:web:d38c0ecd7b1343cc8edadf",
    #"measurementId": "G-P8HMBXN7R4"
    "databaseURL": "https://searchndtry-f0f18-default-rtdb.firebaseio.com/"
}

# init firebase db
firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
db = firebase.database()

# global person dict
person = {"is_logged_in": False, "name": "", "email": "", "uid": ""}

web_det = {}
up_img_path = ""

# login page
@app.route("/")
def login():
    return render_template("login.html")

# signup page
@app.route("/signup")
def signup():
    return render_template("signup.html")

# welcome page
@app.route("/welcome")
def welcome():
    if person["is_logged_in"] == True:
        return render_template("welcome.html", email = person["email"], name = person["name"])
    else:
        return redirect(url_for('login'))

# POST req for login "/result"
@app.route("/result", methods = ["POST", "GET"])
def result():
    if request.method == "POST":        #Only if data has been posted
        result = request.form           #Get the data
        email = result["email"]
        password = result["pass"]
        try:
            #Try signing in the user with the given information
            user = auth.sign_in_with_email_and_password(email, password)
            #Insert the user data in the global person
            global person
            person["is_logged_in"] = True
            person["email"] = user["email"]
            person["uid"] = user["localId"]
            #Get the name of the user
            data = db.child("users").get()
            person["name"] = data.val()[person["uid"]]["name"]
            #Redirect to welcome page
            return redirect(url_for('welcome'))
        except:
            #If there is any error, redirect back to login
            return redirect(url_for('login'))
    else:
        if person["is_logged_in"] == True:
            return redirect(url_for('welcome'))
        else:
            return redirect(url_for('login'))

# POST req for signup "/register"
@app.route("/register", methods = ["POST", "GET"])
def register():
    if request.method == "POST":        #Only listen to POST
        result = request.form           #Get the data submitted
        email = result["email"]
        password = result["pass"]
        name = result["name"]
        try:
            #Try creating the user account using the provided data
            auth.create_user_with_email_and_password(email, password)
            #Login the user
            user = auth.sign_in_with_email_and_password(email, password)
            #Add data to global person
            global person
            person["is_logged_in"] = True
            person["email"] = user["email"]
            person["uid"] = user["localId"]
            person["name"] = name
            #Append data to the firebase realtime database
            data = {"name": name, "email": email}
            db.child("users").child(person["uid"]).set(data)
            #Go to welcome page
            return redirect(url_for('welcome'))
        except:
            #If there is any error, redirect to register
            return redirect(url_for('register'))

    else:
        if person["is_logged_in"] == True:
            return redirect(url_for('welcome'))
        else:
            return redirect(url_for('register'))

# POST req for "add to cart"
@app.route("/cart", methods = ["POST", "GET"])
def cart():
    if request.method == "POST":
        imgUrl = request.form["imgUrl"]
        try:
            db.child("users").child(person["uid"]).child("images").push(imgUrl)
            # return redirect(url_for('cart'))
            global web_det
            global up_img_path
            return render_template('index.html', 
            uploaded_img_path = up_img_path,
            web_detection = web_det)
        except:
            return redirect(url_for('login'))
    else:
        if person["is_logged_in"] == True:
            data = db.child("users").get()
            imgs = data.val()[person["uid"]]["images"]
            return render_template("cart.html", email = person["email"], name = person["name"], imgs = imgs)
        else:
            return redirect(url_for('login'))


@app.route("/cartremove", methods = ["POST", "GET"])
def cartremove():
    if request.method == "POST":
        key = request.form["remove"]
        db.child("users").child(person["uid"]).child("images").child(key).remove()
        if person["is_logged_in"] == True:
            data = db.child("users").get()
            imgs = data.val()[person["uid"]]["images"]
            return render_template("cart.html", email = person["email"], name = person["name"], imgs = imgs)
        else:
            return redirect(url_for('login'))


#------------------------------------------------------------------------------------------firebase auth 

global capture, switch
capture = 0
switch = 0


camera = cv2.VideoCapture(0) 
camera.release()


@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image C:/Users/Rishav/OneDrive/Desktop/cbir/ImageSearch/
        img = Image.open(file.stream) 
        uploaded_img_path = "static/uploads/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        with io.open(uploaded_img_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content = content)
        response = client.web_detection(image = image)
        web_detection = response.web_detection

        global up_img_path 
        up_img_path = uploaded_img_path

        global web_det
        web_det = web_detection

        return render_template('index.html', 
        uploaded_img_path = uploaded_img_path,
        web_detection = web_detection)

    else:
        return render_template('index.html')
 

def gen_frames():  # generate frame by frame from camera
    global capture
    while True:
        success, frame = camera.read() 
        if success:
            if(capture):
                capture=0
                now = datetime.now()
                p = os.path.sep.join(['static/uploads/', "shot_{}.jpg".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            except Exception as e:
                pass
            
        else:
            pass


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1

        elif request.form.get('use') == 'UseImage':
            
            list_of_files = glob.glob('./static/uploads/*.jpg') # specific format *.jpg
            latest_file = max(list_of_files, key=os.path.getctime)
            
            #img = Image.open(latest_file)
            # Run search
            with io.open(latest_file, 'rb') as image_file:
                content = image_file.read()

            image = vision.Image(content = content)
            response = client.web_detection(image = image)
            web_detection = response.web_detection
            
            return render_template('index.html', 
            uploaded_img_path = latest_file,
            web_detection = web_detection)
      
    elif request.method=='GET':
        return render_template('index.html')

    return render_template('index.html')

#----------------------------------------------------------------------------------------vision api

@app.route('/virtualtry', methods=['GET', 'POST'])
def virtualtry():
    if request.method == 'POST':
        imgUrl = request.form["imgUrl"]

        # Save query image 
        urllib.request.urlretrieve(imgUrl, "static/uploads/tryoncloth.jpg")
        uploaded_img_path = "static/uploads/tryoncloth.jpg" 
        
        return render_template('virtualtry.html', 
        uploaded_img_path = uploaded_img_path)

    else:
        return render_template('virtualtry.html')


@app.route('/tryon', methods=['GET', 'POST'])
def tryon():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image 
        img = Image.open(file.stream) 
        uploaded_img_path = "static/uploads/user.jpg" 
        img.save(uploaded_img_path)

        # trying on the cloth and returning final image

        # loading model for virtual try
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

        saved = load_model("save_ckp_frozen.h5")

        class fashion_tools(object):
            def __init__(self,imageid,model,version=1.1):
                self.imageid = imageid
                self.model   = model
                self.version = version
                
            def get_dress(self,stack=False):
                """limited to top wear and full body dresses (wild and studio working)"""
                """takes input rgb----> return PNG"""
                name =  self.imageid
                file = cv2.imread(name)
                # file = tf.image.resize_with_pad(file,target_height=512,target_width=512)
                # if tf.executing_eagerly():
                #     rgb = file.numpy()
                # else:
                #     rgb = file.eval()
                # tf.config.run_functions_eagerly(True)
                file = cv2.resize(file, (512,512))
                # print(type(file))
                rgb  = file   
                #rgb  = file.numpy()
                file = np.expand_dims(file,axis=0)/ 255.
                seq = self.model.predict(file)
                seq = seq[3][0,:,:,0]
                seq = np.expand_dims(seq,axis=-1)
                c1x = rgb*seq
                return c1x
                
                
            def get_patch(self):
                return None

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        model = Model("checkpoints/jpp.pb",
                    "checkpoints/gmm.pth",
                    "checkpoints/tom.pth",
                    use_cuda=False)

        # extracting cloth from image
        api  = fashion_tools('static/uploads/tryoncloth.jpg',saved)
        img = api.get_dress(stack=True)
        cv2.imwrite("static/cloth-old.jpg", img)

        change_bg = alter_bg()
        change_bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
        output = change_bg.color_bg("static/cloth-old.jpg", colors = (255, 255, 255))

        cv2.imwrite("static/cloth.jpg", output)

        # wrapping cloth on user image: uploaded_img_path
        change_bg = alter_bg()
        change_bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
        output = change_bg.color_bg("static/uploads/user.jpg", colors = (255, 255, 255))
        cv2.imwrite("static/person.jpg", output)

        img = Image.open("static/person.jpg")
        w, h = img.size
        img = img.resize((192, 256))
        img = np.array(img)

        c_img = Image.open("static/cloth.jpg")# using extracted cloth-------------------
        c_img = c_img.resize((192, 256))
        #c_img = c_img.convert('RGB')
        c_img = np.array(c_img)

        start = time.time()
        result,trusts = model.predict(img, c_img, need_pre=False,check_dirty=True)
        if result is not None:
            end = time.time()
            print("time:"+str(end-start))
            print("Confidence"+str(trusts))
            #result = cv2.resize(result, (w, h))
            cv2.imwrite("static/final.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            final_img_path = "static/final.jpg"


        return render_template('virtualtry.html', final_img_path = final_img_path)

    else:
        return render_template('index.html')


@app.route('/capturetry',methods=['POST','GET'])
def capturetry():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1

        elif request.form.get('use') == 'UseImage':
            
            list_of_files = glob.glob('./static/uploads/*.jpg') # specific format *.jpg
            latest_file = max(list_of_files, key=os.path.getctime) # returns file path
            
            img = Image.open(latest_file)
            uploaded_img_path = "static/uploads/user.jpg" 
            img.save(uploaded_img_path)

            # trying on the cloth and returning final image

            # loading model for virtual try
            config = ConfigProto()
            config.gpu_options.allow_growth = True
            session = InteractiveSession(config=config)

            saved = load_model("save_ckp_frozen.h5")

            class fashion_tools(object):
                def __init__(self,imageid,model,version=1.1):
                    self.imageid = imageid
                    self.model   = model
                    self.version = version
                    
                def get_dress(self,stack=False):
                    """limited to top wear and full body dresses (wild and studio working)"""
                    """takes input rgb----> return PNG"""
                    name =  self.imageid
                    file = cv2.imread(name)
                    # file = tf.image.resize_with_pad(file,target_height=512,target_width=512)
                    # if tf.executing_eagerly():
                    #     rgb = file.numpy()
                    # else:
                    #     rgb = file.eval()
                    # tf.config.run_functions_eagerly(True)
                    file = cv2.resize(file, (512,512))
                    # print(type(file))
                    rgb  = file   
                    #rgb  = file.numpy()
                    file = np.expand_dims(file,axis=0)/ 255.
                    seq = self.model.predict(file)
                    seq = seq[3][0,:,:,0]
                    seq = np.expand_dims(seq,axis=-1)
                    c1x = rgb*seq
                    return c1x
                    
                    
                def get_patch(self):
                    return None

            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

            model = Model("checkpoints/jpp.pb",
                        "checkpoints/gmm.pth",
                        "checkpoints/tom.pth",
                        use_cuda=False)

            # extracting cloth from image
            api  = fashion_tools('static/uploads/tryoncloth.jpg',saved)
            img = api.get_dress(stack=True)
            cv2.imwrite("static/cloth-old.jpg", img)

            change_bg = alter_bg()
            change_bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
            output = change_bg.color_bg("static/cloth-old.jpg", colors = (255, 255, 255))

            cv2.imwrite("static/cloth.jpg", output)

            # wrapping cloth on user image: uploaded_img_path
            change_bg = alter_bg()
            change_bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
            output = change_bg.color_bg("static/uploads/user.jpg", colors = (255, 255, 255))
            cv2.imwrite("static/person.jpg", output)

            img = Image.open("static/person.jpg")
            w, h = img.size
            img = img.resize((192, 256))
            img = np.array(img)

            c_img = Image.open("static/cloth__.jpg")# using extracted cloth-------------------
            c_img = c_img.resize((192, 256))
            #c_img = c_img.convert('RGB')
            c_img = np.array(c_img)

            start = time.time()
            result,trusts = model.predict(img, c_img, need_pre=False,check_dirty=True)
            if result is not None:
                end = time.time()
                print("time:"+str(end-start))
                print("Confidence"+str(trusts))
                #result = cv2.resize(result, (w, h))
                cv2.imwrite("static/final.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                final_img_path = "static/final.jpg"


            return render_template('virtualtry.html', final_img_path = final_img_path)

        else:
            return render_template('virtualtry.html')
      
    elif request.method=='GET':
        return render_template('virtualtry.html')

    return render_template('virtualtry.html')


@app.route('/temp',methods=['POST','GET'])
def temp():
    return render_template("temp.html")

#----------------------------------------------------------------------------------------virtual try

# run app
if __name__ == "__main__":
    app.run()

camera.release()
cv2.destroyAllWindows() 
 