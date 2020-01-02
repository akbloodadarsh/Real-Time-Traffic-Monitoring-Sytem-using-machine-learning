import cv2 as cv  # Importing computer vision as cv
import numpy as np  # Importing numpy for array processing

confThreshold = 0.25
'''Confidence threshold:-Confidence is a numerical value that is assigned to each Label while Predict is evaluating an
Issue. If the highest Confidence value exceeds the Confidence Threshold as defined by your team then the Label 
associated with that Confidence value is assigned to the Issue.
'''
nmsThreshold = 0.40
'''
Non-maximum suppression:-It will go through all prediction boxes[pc(prediction),bx,by,bh,bw(boxdimensions)] and discard
all boxes which have prediction<= limit, and the from remaining boxes pick up with largest prediction and discard the 
remaining with intersection over union value which exceeds limit
'''
'''
Yolo only accepts 3 size:-320*320 fast speed,less accuracy,609*609 high accuracy but slow speed 416*416 its middle'''
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

# Load names of classes and turn that into a list
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
'''from file it Splits at '\n' then return a list then take the list 
and removing \n from the string using rstrip('\n')
'''

modelConf = 'yolov3.cfg'  # Configuration File:-: it’s the configuration file, where there are all the settings
modelWeights = 'yolov3.weights'  # Pre-trained Weights:-it’s the trained model to detect the objects.


def postprocess(frame, outs, prevcount):
    # get height of frame
    frameHeight = frame.shape[0]
    # get width of frame
    frameWidth = frame.shape[1]
    # create a empty list for ids of class
    classIDs = []
    # create a empty list for confidence of class
    confidences = []
    # create a empty list for boxes of class
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confThreshold:
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)

                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)

                left = int(centerX - width / 2)
                top = int(centerY - height / 2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    # set a initial count of vehicles
    count = 0

    for i in indices:

        # stores the index
        i = i[0]

        # stores the dimensions of box (x,y,w,h)
        box = boxes[i]

        # stores the x
        left = box[0]

        # stores the y
        top = box[1]

        # stores width
        width = box[2]

        # stores height
        height = box[3]

        if classIDs[i] == 2 or classIDs[i] == 1 or classIDs[i] == 5 or classIDs[i] == 7:
            count = count + 1
        drawPred(classIDs[i], confidences[i], left, top, left + width, top + height)

    # if previous count is different than current count then print & store count in previous count
    if prevcount != count:
        print("Number of vehicles=", count)
        prevcount = count
    return prevcount


def drawPred(classId, conf, left, top, right, bottom):
    # if any cond gets true draw bounding boxes(1=bicycle,2=car,3=motorbike,5=bus,7=truck)
    if classId == 1 or classId == 2 or classId == 3 or classId == 5 or classId == 7:
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        label = '%.2f' % conf
        # Get the label for the class name and its confidence
        if classes:
            assert (classId < len(classes))
            label = '%s:%s' % (classes[classId], label)
        '''
        # A fancier display of the label from learnopencv.com
        # Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[0])
        cv.rectangle(frame, (left, top - round(1.5 * labelSize[0])), (left + round(1.5 * labelSize[0]), top + baseLine),cv.FILLED)
        # cv.rectangle(frame, (left,top),(right,bottom), (255,255,255), 1 )
        # cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
        cv.putText(frame, label, (left,top), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        '''


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Set up the net
net = cv.dnn.readNetFromDarknet(modelConf, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# To set the name of window
winName = 'Window'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
# To set size of video o/p window
cv.resizeWindow(winName, 480, 480)

# To import video
cap = cv.VideoCapture('cars.mp4')
# seting the initial previous count
prevcount = 0
while cv.waitKey(1) < 0:
    # get frame from video
    hasFrame, frame = cap.read()
    # Create a 4D blob from a frame
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    # Set the input the the net
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))
    prevcount = postprocess(frame, outs, prevcount)
    # show the frame
    cv.imshow(winName, frame)
