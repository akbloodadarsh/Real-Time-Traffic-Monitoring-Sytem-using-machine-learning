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

# Yolo only accepts 3 size:-320*320 fast speed,less accuracy,609*609 high accuracy but slow speed 416*416 its middle
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

# Load names of classes and turn that into a list
# coco(common objects in context)
classesFile = "coco.names"
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
'''from file it Splits at '\n' then return a list then take the list 
and removing \n from the string using rstrip('\n')
'''

modelConf = 'yolov3.cfg'  # Configuration File:-: it’s the configuration file, where there are all the settings
modelWeights = 'yolov3.weights'  # Pre-trained Weights:-it’s the trained model to detect the objects.


def postprocess(frame, p_frame, v_frame, outs, prev_veh_count, prev_ped_count):
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
            # detection will contain all the attributes of objects detected like x,y,w,h,confidence etc
            # start at 5th index to the end then store all in scores
            scores = detection[5:]
            # argmax:-return Array of indices into the array with same shape as array.shape with the dimension along
            # axis removed.
            # classID:-stores all the class id
            classID = np.argmax(scores)
            # stores all the confidence of the classID
            confidence = scores[classID]
            if confidence > confThreshold:
                # detection[0]:- contains x
                # it will store x of the actual frame
                centerX = int(detection[0] * frameWidth)
                # detection[1]:- contains y
                # it will store y of the actual frame
                centerY = int(detection[1] * frameHeight)
                # detection[2]:- contains width
                # it will store w of the actual frame
                width = int(detection[2] * frameWidth)
                # detection[3]:- contains height
                # it will store h of the actual frame
                height = int(detection[3] * frameHeight)

                # For adjustment
                # it will give actual position from left
                left = int(centerX - width / 2)
                # it will give actual position from top
                top = int(centerY - height / 2)

                # stores the classID in classIDs list
                classIDs.append(classID)
                # stores the confidence in confidences list
                confidences.append(float(confidence))
                # stores the dimensions in boxes list
                boxes.append([left, top, width, height])

    # it will suppress all the extra boxes that are detected and only returns with the high confidence and accuracy
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    # set a initial count of vehicles
    count = 0
    p_count = 0
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
        if classIDs[i] == 0:
            p_count = p_count + 1
        v_frame, p_frame = drawPred(classIDs[i], left, top, left + width, top + height, v_frame, p_frame)
    # if previous count is different than current count then print & store count in previous count
    if prev_veh_count != count:
        print("Number of vehicles=", count)
        prev_veh_count = count
    if prev_ped_count != p_count:
        print("Number of pedestrains=", p_count)
        prev_ped_count = p_count
    return prev_veh_count, prev_ped_count, p_frame, v_frame


def drawPred(classId, left, top, right, bottom, v_frame, p_frame):
    # if any cond gets true draw bounding boxes(1=bicycle,2=car,3=motorbike,5=bus,7=truck)
    if classId == 1 or classId == 2 or classId == 3 or classId == 5 or classId == 7:
        # Draw a bounding box.
        v_frame = cv.rectangle(v_frame, (left, top), (right, bottom), (255, 178, 50), 3)
    elif classId == 0:
        # Draw a bounding box.
        p_frame = cv.rectangle(p_frame, (left, top), (right, bottom), (255, 178, 50), 3)
    return v_frame, p_frame


'''Generally in a sequential CNN network there will be only one output layer at the end. In the YOLO v3 architecture we
there are multiple output layers giving out predictions. get_output_layers() function gives the names of the output
layers. An output layer is not connected to any next layer
'''


def getOutputsNames(net):
    # Get the names of all the layers in the network
    """
    Convulation:-we have a input,such as an image of pixel values,& we have a filter,which is a set of weights,& the fi-
    ter is systematically applied to the i/p data to create a feature map(o/p of one filter applied to previous layer)

    Batch Normalization:-reduce covariate shift(distribution b/w layers),fix mean=0 and variance=1,stable distribution

    A Neural Network without Activation function would simply be a Linear regression Model
    without activation function our Neural network would not be able to learn and model other complicated kinds of data
    such as images, videos , audio , speech etc
    non linear Activation we are able to generate non-linear mappings from inputs to outputs.Ex:-
    (i)Sigmoid or Logistic(Vanishing gradient problem,not zero centered,(0<o/p<1))formula 1/(1+exp^(-x)) where x is i/p
    It maps and contains only values from 0 to 1, the more chances of having 1 the more chance of activation
    (ii)Tanh — Hyperbolic tangent(Vanishing gradient problem,zero centered,(-1<o/p<1))form:- [2/(1+e^(-2x))]-1
    (iii)ReLu -Rectified linear units(only be used within Hidden layers,dying neurons) #Currently used
    if the i/p is close to 0 ex:--2 it will set it to
    0,and the more closer it will be to 1 it will set it to 1.Removes the negative part of function & let the +ve remain
    if suffers from dead neurons, leaky ReLu, maxout(ReLu+leaky ReLu)

    Shortcut:-This architecture can concatenate multi-scale feature maps by shortcut connections to form the fully-
    connected layer that is directly fed to the o/p layer
    """
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    # getUnconnectedOutLayers() Returns indexes of layers with unconnected outputs.
    # layerNames contains the name of the unconnected layers
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def filtering(fil_frame):
    # replace each pixel value with the median of its neighbouring pixels
    # k - size
    fil_frame = cv.medianBlur(fil_frame, 3)
    # non-linear,preserve border & edge,noise-reducing smoothing.replaces intensity
    # of each pixel with weighted avg of intensity values from nearby pixels.
    # diameter of each pixel neighbourhood, sigma color
    fil_frame = cv.bilateralFilter(fil_frame, 3, 75, 75)
    return fil_frame


# *****MAIN***** #
print("Press 1 to save the video\nPress any other key to continue without saving\n")
choice = str(input())

# read the weights and configuration and creates the network
net = cv.dnn.readNetFromDarknet(modelConf, modelWeights)
# Ask network to use specific computation backend where it supported
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# Ask network to make computations on specific target device
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# To set the name of window
v_window = 'vehicles'
p_window = 'pedestrians'
cv.namedWindow(v_window, cv.WINDOW_NORMAL)
# To set size of video o/p window
cv.resizeWindow(v_window, inpWidth, inpHeight)
cv.resizeWindow(p_window, inpWidth, inpHeight)

# To import video
cap = cv.VideoCapture('test1')
v_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
v_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# For having a constant size in every resolution
scale = 0.05
fontScale = min(v_width, v_height)/(25/scale)

# For having a constant size of rectangle in every resolution
rec_width = int(v_width/1.8)
rec_height = int(v_height/6.8)

# setting the initial previous count
# For vehicles
prev_veh_count = 0
# For pedestrians
prev_ped_count = 0

if choice == '1':
    # setting codec to encode or decode digital stream
    four_cc = cv.VideoWriter_fourcc(*'XVID')

    # to save video
    save_ped = cv.VideoWriter('vid_p.avi', four_cc, 15, (v_width, v_height))
    save_veh = cv.VideoWriter('vid_v.avi', four_cc, 15, (v_width, v_height))

    while True:
        # get frame from video
        ret, frame = cap.read()
        # applying filter
        frame = filtering(frame)
        p_frame = frame
        v_frame = frame
        if ret:
            # Create a 4D blob from a frame cv.dnn.blobFromImage(image,scale_factor,size,mean, swapRB,crop,depth)
            '''a blob is a group of connected pixels,neighbouring pixels'''
            blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
            # Set the i/p blob
            net.setInput(blob)
            # outs:-Array that contains all info about objects detected,their position,confidence about the detection
            # forward:- (blob already set)compute o/p of all the layers and returns the blob
            outs = net.forward(getOutputsNames(net))
            # send the frame, detected obj,previous count and return the updated previous count
            prev_veh_count, prev_ped_count, p_frame, v_frame = postprocess(frame, p_frame, v_frame, outs, prev_veh_count
                                                                           , prev_ped_count)
            v_text = "Number of vehicles=" + str(prev_veh_count)
            v_frame = cv.rectangle(v_frame, (0, 0), (rec_width, rec_height), (0, 0, 0), -1)
            # 0.8
            v_frame = cv.putText(v_frame, v_text, (10, 30), cv.FONT_HERSHEY_COMPLEX, fontScale, (0, 0, 255), 0, cv.LINE_AA)
            # show the frame
            cv.imshow(v_window, v_frame)
            save_veh.write(v_frame)
            p_text = "Number of pedestrians=" + str(prev_ped_count)
            p_frame = cv.rectangle(p_frame, (0, 0), (rec_width, rec_height), (0, 0, 0), -1)
            p_frame = cv.putText(p_frame, p_text, (10, 30), cv.FONT_HERSHEY_COMPLEX, fontScale, (0, 0, 255), 0, cv.LINE_AA)
            save_ped.write(p_frame)
            cv.imshow(p_window, p_frame)
            if cv.waitKey(1) & 0XFF == ord('q'):
                break
    save_ped.release()
    save_veh.release()
else:
    while True:
        # get frame from video
        ret, frame = cap.read()
        frame = filtering(frame)
        p_frame = frame
        v_frame = frame
        if ret:
            # Create a 4D blob from a frame cv.dnn.blobFromImage(image,scale_factor,size,mean, swapRB,crop,depth)
            '''a blob is a group of connected pixels,neighbouring pixels'''
            blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
            # Set the i/p blob
            net.setInput(blob)
            # outs:-Array that contains all info about objects detected,their position,confidence about the detection
            # forward:- (blob already set)compute o/p of all the layers and returns the blob
            outs = net.forward(getOutputsNames(net))
            # send the frame, detected obj,previous count and return the updated previous count
            prev_veh_count, prev_ped_count, p_frame, v_frame = postprocess(frame, p_frame, v_frame, outs, prev_veh_count
                                                                           , prev_ped_count)

            # For vehicles
            v_text = "Number of vehicles=" + str(prev_veh_count)
            v_frame = cv.rectangle(v_frame, (0, 0), (rec_width, rec_height), (0, 0, 0), -1)
            v_frame = cv.putText(v_frame, v_text, (10, 30), cv.FONT_HERSHEY_COMPLEX, fontScale, (255, 255, 255), 0)
            # show the frame
            cv.imshow(v_window, v_frame)

            # For pedestrians
            p_text = "Number of pedestrians=" + str(prev_ped_count)
            p_frame = cv.rectangle(p_frame, (0, 0), (rec_width, rec_height), (0, 0, 0), -1)
            p_frame = cv.putText(p_frame, p_text, (10, 30), cv.FONT_HERSHEY_PLAIN, fontScale, (255, 255, 255), 0)
            # show the frame
            cv.imshow(p_window, p_frame)
            if cv.waitKey(1) & 0XFF == ord('q'):
                break
cv.destroyAllWindows()
