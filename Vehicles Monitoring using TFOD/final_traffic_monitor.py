"""Loading the required libraries """
import cv2
import numpy as np
import datetime
from utils import detector_utils as detector_utils
from utils import visualization_utils as vis_util
from utils import label_map_util
print('importing done') 


#Selecting the model file
TRAINED_MODEL_DIR = 'E:\AI\Projects\Traffic_Monitoring\Actual_app/frozen_graphs'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = TRAINED_MODEL_DIR + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = TRAINED_MODEL_DIR + '/mscoco_label_map.pbtxt'

NUM_CLASSES = 90
# load label map using utils provided by tensorflow object detection api
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#Loading the model
detection_graph, sess = detector_utils.load_inference_graph()

#Loading the video input
video_path = 'traffic_vid_2.mp4'
# Check if camera opened successfully
cap = cv2.VideoCapture(video_path)
if (cap.isOpened()== False):
  print("Error opening video stream or file")
    
# Used to calculate fps
start_time = datetime.datetime.now()
num_frames = 0
 
# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        ret, frame = cap.read()
        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run image through tensorflow graph
        boxes, scores, classes = detector_utils.detect_objects(frame, detection_graph, sess)
        
        #calulate the FPS
        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time

        #Visualize boxes and labels
        vis_util.visualize_boxes_and_labels_on_image_array( frame,boxes, (np.int32(np.squeeze(classes))),scores,category_index, use_normalized_coordinates= True, line_thickness= 3)
        
        #Draw FPS on frame
        detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)
 
        # Display the resulting frame
        height, width = frame.shape[:2]
        height = int(height*0.5)
        width = int(width *0.5)
        #cv2.namedWindow('Detected Output', cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('Detected Output', width,height)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Detected Output', frame)
        
        # Press 'Q' on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()