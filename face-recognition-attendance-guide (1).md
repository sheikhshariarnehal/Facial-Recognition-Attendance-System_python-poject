# Face Recognition Based Attendance System: Step-by-Step Guide

## Prerequisites
1. Python 3.7+ installed
2. Visual Studio Code installed
3. Webcam (built-in or external)

## Step 1: Set up the project

1. Open Visual Studio Code.
2. Create a new folder for your project.
3. Open the terminal in VS Code (View > Terminal).
4. Navigate to your project folder.

## Step 2: Set up a virtual environment

1. In the terminal, create a virtual environment:
   ```
   python -m venv venv
   ```
2. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

## Step 3: Install required packages

Install the following packages using pip:

```
pip install flask opencv-python numpy scikit-learn pandas joblib
```

## Step 4: Create project files

1. Create a new file named `app.py` and copy the provided Python code into it.
2. Create a new folder named `templates` in your project directory.
3. Inside the `templates` folder, create a file named `home.html` and copy the provided HTML code into it.
4. Download the `haarcascade_frontalface_default.xml` file from the OpenCV GitHub repository and place it in your project folder.
5. Create an empty folder named `static` in your project directory.
6. Inside the `static` folder, create an empty folder named `faces`.
7. Download a background image (e.g., from a free stock photo site) and save it as `background.png` in your project folder.

## Step 5: Modify the code (if necessary)

1. In `app.py`, check if the following line matches your background image filename:
   ```python
   imgBackground = cv2.imread("background.png")
   ```
2. If you want to change the number of images captured per user, modify the `nimgs` variable:
   ```python
   nimgs = 10
   ```

## Step 6: Run the application

1. In the VS Code terminal, make sure your virtual environment is activated.
2. Run the following command:
   ```
   python app.py
   ```
3. Open a web browser and go to `http://127.0.0.1:5000/`

## Step 7: Use the application

1. To add a new user:
   - Click on "Add New User"
   - Enter the user's name and ID
   - Click "Add New User"
   - Follow the on-screen instructions to capture face images

2. To take attendance:
   - Click on "Take Attendance"
   - The webcam will open and recognize faces
   - Press 'Esc' to stop the attendance process

3. View attendance:
   - The main page will display today's attendance

## Troubleshooting

- If you encounter any errors related to missing modules, make sure all required packages are installed in your virtual environment.
- If the webcam doesn't open, check your webcam connections and permissions.
- If face recognition is not working properly, try adjusting lighting conditions or camera position.

## Next Steps

- Implement user authentication for the web interface
- Add a database to store user information and attendance records
- Improve the UI design using CSS frameworks like Bootstrap
- Implement more robust face recognition algorithms (e.g., using deep learning)
