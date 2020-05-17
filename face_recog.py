import face_recognition
from PIL import Image,ImageDraw

#from IPython.display import Image 
#pil_img = Image(filename='/content/Cristiano-Ronaldo2.jpg')
#display(pil_img)

image_of_ronaldo = face_recognition.load_image_file('/content/Cristiano-Ronaldo2.jpg')
ronaldo_face_encoding = face_recognition.face_encodings(image_of_ronaldo)[0]


image_of_messi = face_recognition.load_image_file('/content/messi.jpeg')
messi_face_encoding = face_recognition.face_encodings(image_of_messi)[0]


known_face_encodings=[ronaldo_face_encoding,messi_face_encoding]

known_face_names =["ronaldo","messi"]


test_image= face_recognition.load_image_file('/content/3_legends.jpg')

face_locations =face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image,face_locations)
pil_image= Image.fromarray(test_image)
draw = ImageDraw.Draw(pil_image)
for(top,right,bottom,left),face_encoding in zip(face_locations,face_encodings):
  matches= face_recognition.compare_faces(known_face_encodings,face_encoding)

  name = "unknown"
  #if match
  if True in matches:
     first_match_index =matches.index(True)
     name = known_face_names[first_match_index]
  
  draw.rectangle(((left,top),(right,bottom)),outline=(0,255,0))

  text_width,text_height = draw.textsize(name)
  draw.rectangle(((left,bottom -text_height -10),(right,bottom)),fill=(0,0,0),outline=(0,255,0))
  draw.text((left+ 6 ,bottom - text_height -5),name, fill=(255,255,255,255))

del draw

from IPython.display import Image 
pil_image_by_colab=pil_image
display(pil_image_by_colab)
#pil_image.show()

#print('worked')
