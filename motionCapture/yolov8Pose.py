import cv2
import time
from ultralytics import YOLO

print('Starting...')

model = YOLO('yolov8n-pose.pt')


# 동영상 파일 사용시
# video_path = 'C:\_data\project\dance\\NMIXX__O.O_.mp4'
# cap = cv2.VideoCapture(video_path)

# webcam 사용시
cap = cv2.VideoCapture(0)


start_time = time.time()
frame_count = 0
results = None

count = 0

while cap.isOpened():
  # Read a frame from the video
  success, frame = cap.read()

  count = count + 1
 
  if success:
      # Resize the image to the desired size using cv2.resize()
      # frame = cv2.resize(frame, (width, height))

      # Check if one second has passed since the last inference was applied
      if time.time() - start_time >= 0.5:
          # Run YOLOv8 inference on the frame
          results = model(frame)

          # Update the timer and frame counter
          start_time = time.time()
          #frame_count += 2

      # Visualize the results on the frame if they exist
      if results is not None:
          annotated_frame = results[0].plot()

          # Display the annotated frame
          cv2.imshow("YOLOv8 Inference", annotated_frame)


          P = 0
          try:
              # print keypoints index number and x,y coordinates
              for idx, kpt in enumerate(results[0].keypoints[0]):
                  print('Persons Detected')
                  P = 1
          except:
                  print('No Persons')
                  P = 0

      # Break the loop if 'q' is pressed
      if cv2.waitKey(1) & 0xFF == ord("q"):
          break
  else:
      # Break the loop if the end of the video is reached
      break