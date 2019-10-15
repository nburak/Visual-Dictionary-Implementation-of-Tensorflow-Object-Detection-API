# Visual Dictionary : Implementation of Tensorflow Object Detection API
Tensorflow object detection API was used to create this application.

<h3>Very Important!</h3>
Full source code can be reached using release tab above. Also main.py file can be used for referencing.

<h3>What it does?</h3>
It provides a web page and users can upload images using this web page. After clicking on search button, it will return translation of the object in the picture for different languages. (English, Spanish, German, French and Turkish)

<h3>Which model was used?</h3>
It uses pre-trained SSD model by Google and this model was trained using coco dataset which includes only 91 different object classes. However, you may train your own model with a dataset which includes support of more classes.

<h3>Can another neural network model be used?</h3>
Yes, you can train your own model. So, software will be able to detect much more different object class. This is for referencing and standard SSD model of Google was used.

<h3>Example</h3>
<p>Used Image</p>
<img src="https://github.com/nburak/Visual-Dictionary-Implementation-of-Tensorflow-Object-Detection-API/blob/master/00003.jpg?raw=true" width="400px" height="auto">
<p>Result</p>
<img src="https://github.com/nburak/Visual-Dictionary-Implementation-of-Tensorflow-Object-Detection-API/blob/master/result.png?raw=true" width="400px" height="auto"  border="5">

<h3>Used Technologies</h3>
<p>Python and Flask were chosen as programming language and development server.</p>
<p>Tensorflow object detection API was chosen for object detection.</p>
<p>Google translate api was chosen for translation of detected objects.</p>
