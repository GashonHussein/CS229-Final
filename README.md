# Drowsiness Detection
Computer Vision based drowsiness detection problem. Utilizes 50,000+ parameter logistic regression baseline model and a 3 layer fully-connected convolutional neural network with padding and various kernels + hyperparameter tuning. Best 3 layer model achieved a testing accuracy of 99.8% when classifying people as drowsy or non-drowsy.
# Instructions

## Create the isolated env
`python3 -m venv env`
`source env/bin/activate`

## If you add new dependencies
`pip3 install -r requirements.txt`

## Write code to handle each image in the database
Edit code in /util/process_image

## Testing
`python3 main.py`
