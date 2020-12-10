# Handwritten-Digit-Recognition
The project is done in a few steps:
1. The training and testing datasets are loaded from the mnist dataset by keras. Then those data images are updated to fit the requirements of our model during training.
2. One-hot encoding is done on the dependent variable, and a column is shaped for every type of output and offered with a binary value
3. CNN sequential model is formed and assemled
4. The model is then trained and the trained weights are stored in a file called 'mnist.h5'. An H5 file is a data file saved in the Hierarchical Data Format (HDF). It contains multidimensional arrays of scientific data.
5. An cooperative window formed using python Tkinter library, to draw digits on canvas and buttons to  forecast output and clear screen.
6. When the forecast button is pushed, handle related to canvas window is kept in HWND. Then the image of canvas screen is grabbed with the help of PIL ImageGrab. This image is passed to predict_digit() function to predict digit.
7. 
