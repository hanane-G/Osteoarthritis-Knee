from flask import Flask, render_template, request
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

# Initialiser l'application 
app = Flask(__name__)


#dic = {0 : 'Normal', 1 : 'Doubtful', 2 : 'Mild', 3 : 'Moderate', 4 : 'Severe'}
dic = {0 : 'Normal', 1 : 'Douteux', 2 : 'Léger', 3 : 'Modéré', 4 : 'Sévère'}

#Image Size
img_size=256

#charger le modèle et creer la fonction de prediction
model = load_model('model.h5')

#model.make_predict_function()


#fct qui prend en entrée l'image et renvoies une prediction du modele sur l'image

def predict_label(img_path):
    #cgarger l'image à partir du chemin specifié
    img=cv2.imread(img_path)
    #convertir l'image en niveau de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Redimensionner l'image   
    resized=cv2.resize(gray,(img_size,img_size))
    #convertir l'image en un tableau numpy et normalise les valeurs des pixels de l'image en les divisant par 255.0
    i = image.img_to_array(resized)/255.0
    #remodèler le tableau numpy i pour qu'il ait la forme requise par le modèle
    #Le modèle s'attend à recevoir une entrée sous la forme d'un tableau 4D avec une dimension de lot (batch dimension), ici de taille 1.
    i = i.reshape(1,img_size,img_size,1)
    #utiliser la fonction predict() du modèle
    p = model.predict(i)
    #obtenir la classe prédite ayant la plus haute probabilité
    predicted_class = np.argmax(p, axis=-1)
    return dic[predicted_class[0]]




# routes "/"
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")
#En visitant http://localhost:5000/ dans son navigateur), 
#la fonction main() sera appelée pour générer une réponse HTTP.

#ajouter la page 'patient'
#@app.route("/test.html", methods=['GET', 'POST'])
#def patients():
#    return render_template("test.html")


#route "/predict"
@app.route("/predict", methods = ['GET', 'POST'])
def upload():
    
    if request.method == 'POST':
       img = request.files['file']
       #Extraire le fichier envoyé par l'utilisateur
       img_path = "uploads/" + img.filename    
       #definir le chemin ou le fichier sera enregistrer 
       img.save(img_path)
       #enregistrer le fichier
       p = predict_label(img_path)
       #renvoyer la prediction du modele pour l'image
       print(p)
       return str(p).lower()

if __name__ =='__main__':#Vérifier si ce fichier Python est exécuté directement (et non importé en tant que module dans un autre fichier Python
    #app.debug = True
    app.run(debug = True)
    #lancer l'application en mode debogage : afficher les erreurs dans la console 