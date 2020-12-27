import cv2
#from thread_demo import putIterationsPerSec 
from CountsPerSec import CountsPerSec
from VideoGet import VideoGet
from VideoShow import VideoShow
import math

def putIterationsPerSec(frame, iterations_per_sec):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame

def threadBoth(source=0):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Dedicated thread for showing video frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """

    video_getter = VideoGet(source).start()
    video_shower = VideoShow(video_getter.frame).start()
    cps = CountsPerSec().start()

    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            break

        frame = video_getter.frame
        frame = putIterationsPerSec(frame, cps.countsPerSec())
        video_shower.frame = frame
        cps.increment()

def threadVideoShow(source=0):
    """
    Dedicated thread for showing video frames with VideoShow object.
    Main thread grabs video frames.
    """

    cap = cv2.VideoCapture(source)
    (grabbed, frame) = cap.read()
    video_shower = VideoShow(frame).start()
    cps = CountsPerSec().start()

    while True:
        (grabbed, frame) = cap.read()
        if not grabbed or video_shower.stopped:
            video_shower.stop()
            break

        frame = putIterationsPerSec(frame, cps.countsPerSec())
        video_shower.frame = frame
        cps.increment()

def sort_corners(corner1, corner2, corner3, corner4):
    """
    Sort the corners such that
    - A is top left
    - B is top right
    - C is bottom left
    - D is bottom right
    Return an (A, B, C, D) tuple
    """
    results = []
    corners = (corner1, corner2, corner3, corner4)

    min_x = None
    max_x = None
    min_y = None
    max_y = None

    for (x, y) in corners:
        if min_x is None or x < min_x:
            min_x = x

        if max_x is None or x > max_x:
            max_x = x

        if min_y is None or y < min_y:
            min_y = y

        if max_y is None or y > max_y:
            max_y = y

    # top left
    top_left = None
    top_left_distance = None
    for (x, y) in corners:
        distance = pixel_distance((min_x, min_y), (x, y))
        if top_left_distance is None or distance < top_left_distance:
            top_left = (x, y)
            top_left_distance = distance

    results.append(top_left)

    # top right
    top_right = None
    top_right_distance = None

    for (x, y) in corners:
        if (x, y) in results:
            continue

        distance = pixel_distance((max_x, min_y), (x, y))
        if top_right_distance is None or distance < top_right_distance:
            top_right = (x, y)
            top_right_distance = distance
    results.append(top_right)

    # bottom left
    bottom_left = None
    bottom_left_distance = None

    for (x, y) in corners:
        if (x, y) in results:
            continue
            distance = pixel_distance((min_x, max_y), (x, y))

        if bottom_left_distance is None or distance < bottom_left_distance:
            bottom_left = (x, y)
            bottom_left_distance = distance
    results.append(bottom_left)

    # bottom right
    bottom_right = None
    bottom_right_distance = None

    for (x, y) in corners:
        if (x, y) in results:
            continue

        distance = pixel_distance((max_x, max_y), (x, y))

        if bottom_right_distance is None or distance < bottom_right_distance:
            bottom_right = (x, y)
            bottom_right_distance = distance
    results.append(bottom_right)

    return results


def pixel_distance(A, B):
    """
    Pythagrian therom to find the distance between two pixels
    """
    A = (1,1)
    B = (1,1)
    (col_A, row_A) = A
    (col_B, row_B) = B

    return (math.sqrt(math.pow(col_B - col_A, 2) + math.pow(row_B - row_A, 2))+0)

def approx_is_square(approx, SIDE_VS_SIDE_THRESHOLD=0.70, ANGLE_THRESHOLD=20, ROTATE_THRESHOLD=30):
    """
    Rules
    - there must be four corners
    - all four lines must be roughly the same length
    - all four corners must be roughly 90 degrees
    - AB and CD must be horizontal lines
    - AC and BC must be vertical lines
    SIDE_VS_SIDE_THRESHOLD
        If this is 1 then all 4 sides must be the exact same length.  If it is
        less than one that all sides must be within the percentage length of
        the longest side.
        A ---- B
        |      |
        |      |
        C ---- D
    """

    assert SIDE_VS_SIDE_THRESHOLD >= 0 and SIDE_VS_SIDE_THRESHOLD <= 1, "SIDE_VS_SIDE_THRESHOLD must be between 0 and 1"
    assert ANGLE_THRESHOLD >= 0 and ANGLE_THRESHOLD <= 90, "ANGLE_THRESHOLD must be between 0 and 90"

    # There must be four corners
    if len(approx) != 4:
        return False
    
    # Find the four corners
    (A, B, C, D) = sort_corners(tuple(approx[0][0]),
                                tuple(approx[1][0]),
                                tuple(approx[2][0]),
                                tuple(approx[3][0]))

    # Find the lengths of all four sides
    AB = pixel_distance(A, B)
    AC = pixel_distance(A, C)
    DB = pixel_distance(D, B)
    DC = pixel_distance(D, C)
    distances = (AB, AC, DB, DC)
    max_distance = max(distances)
    cutoff = int(max_distance * SIDE_VS_SIDE_THRESHOLD)

    # If any side is much smaller than the longest side, return False
    for distance in distances:
        if distance < cutoff:
            return False

    return True

def process(img):                                                                                                                                                       
    #Process an bgr image to binary 
    #kernel = np.ones((3,3),np.uint8) this is an alternative way to create kernel
    #img1= cv2.pyrMeanShiftFiltering(img, 5, 40)
    #img1 = cv2.medianBlur(img, 5)
    #img1 = cv2.GaussianBlur(img1, (3, 3), 0)
    blur = cv2.blur(img, (3, 3))
    # 方框滤波（归一化）=均值滤波
    box1 = cv2.boxFilter(img, -1, (3, 3), normalize=True)
    # 方框滤波（不归一化）
    box2 = cv2.boxFilter(img, -1, (3, 3), normalize=False)
    # 高斯滤波
    # 用5*5的核进行卷积操作，但核上离中心像素近的参数大。
    guassian = cv2.GaussianBlur(img, (5, 5), 1)
    # 中值滤波
    # 将某像素点周围5*5的像素点提取出来，排序，取中值写入此像素点。
    img1 = cv2.medianBlur(img, 5)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #Corresponding grayscale image to the input
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,3)
    binary_blurred = cv2.medianBlur(binary,5)
    binary_dilated = cv2.dilate(binary_blurred,kernel,iterations = 8)
    binary_inv = 255 - binary_dilated

    return binary_inv, gray


def threadVideoGet(source=0):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Main thread shows video frames.
    """

    video_getter = VideoGet(source).start()
    cps = CountsPerSec().start()

    while True:
        if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
            video_getter.stop()
            break

        frame = video_getter.frame
        frame = putIterationsPerSec(frame, cps.countsPerSec())
        #cv2.rectangle(frame,(630, 390),(865, 631),(0,255,0),2)
        #frame = cv2.flip(frame,1)  #镜像
        #cube_range = frame[630:390, 865:631]
        cube_range = frame[390-10:630+10, 631-10:865+10]
        img, gray = process(cube_range)

        
        recnum = 0;
        # findContours 会返回两个矩阵
        # contours  包含了每一个轮廓线
        # hierarchy 包含了对应轮廓线的阶级，也就是被多少其他轮廓线包含着
        contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt,0.12*cv2.arcLength(cnt,True),True)
            #cv2.drawContours(cube_range, approx, -1, (0, 0, 255), 3)
            #cv2.polylines(cube_range, [approx], True, (0, 0, 255), 2)
            #cv2.polylines(img, [approx], True, (0, 0, 255), 2)

            x = approx.ravel()[0]
            y = approx.ravel()[1]
            if (len(approx) == 4):
                print("x", x)
                print("y", y)

            if (len(approx) == 4):
                # and 245<x<395 and 105<y<255):
                #Approx has 4 (x,y) coordinates, where the first is the top left,and
                #the third is the bottom right. Findind the mid point of these two coordinates
                #will give me the center of the rectangle
                recnum = recnum + 1 
                    
                x1=approx[0,0,0]
                y1=approx[0,0,1]
                x2=approx[(approx.shape[0]-2),0,0] #X coordinate of the bottom right corner
                y2=approx[(approx.shape[0]-2),0,1] 
                    
                xavg = int((x1+x2)/2)
                yavg = int((y1+y2)/2)

                #cv2.drawContours(img, approx, -1, (0, 0, 255), 3)
                #cv2.polylines(img, [approx], True, (0, 0, 255), 2)

                #if (recnum > 9): 
                   # break

                if (approx_is_square(approx) == True):
                    print("!!!!!!!!!!!!!!11")
                    #cv2.circle(img,(xavg,yavg),15,(255, 0, 0),5)
                    cv2.circle(cube_range,(xavg,yavg),15,(255, 0, 0),5)
                ''' 
                if (recnum == 9 and approx_is_square(approx) == True):
                    string = get_string(cords)
                    color_string = get_average_color(image,string)
                    thecolor = get_color_string(color_string)
                        
                    create_referrence_color(image,thecolor)  
                '''
        print("**********************************88")

        #cv2.imshow("Video", gray)
        #cv2.imshow("Video", img)
        #cv2.imshow("Video", binary_dilated)
        #cv2.imshow("Video", frame)
        cv2.imshow("Video", cube_range)
        cps.increment()

threadVideoGet(source=0)
