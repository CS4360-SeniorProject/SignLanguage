## What you need
- An IDE that can run Jupyter Notepook ipynb files (We reccomend VSCode).
- A Python 3.9 environment

## Initial run instructions
- Clone the repository to your local machine ```git clone https://github.com/CS4360-SeniorProject/SignLanguage.git```
- Navigate into ```SignLanguage > Model Trainer``` subdirectory
- Open ```Model_maker.ipynb``` with your IDE
- Click "Run All"
- Press Q at any time to stop the camera.

  !! Please note: It may take a while to run all the cells for the first time. If you do not already have a Python3 environment, VSCode can set it up for you. This will also take time.
  
## Run with predictions
After the initial run, to see the program make predictions based on the landmark detection of the hand, you can follow these steps.

- Ensure the camera is not currently running. If it is, in the camera window press Q.
- In ```Model_maker.ipynb```, scroll down to Section 3: Extract Keypoint Values
- If using VSCode, you can run individual cells by pressing the "play" button on the left of the cell. It will pop up when hovering over the desired cell.
- Run the cell with ```def extract_keypoints(results):``` and the cell below with ```result_test = extract_keypoints(results)```.
- Scroll to Section 10: Load Models and Labels...
- Run each cell in Section 10.
- Finally, scroll to Section 11: Test in Real Time, and run the first cell.

The program should open another instance of the camera with landmark detection, and should display the predicted letter in the top left corner. The program currently will detect A, B and C in ASL.