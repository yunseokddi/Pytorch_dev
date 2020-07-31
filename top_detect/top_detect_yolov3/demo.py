from top_detect import top_detect

import timeutil

detection = top_detect(weigth_PATH='./weights/yolov3_ckpt_99.pth')  # change your image path and weight path

total_time = 0.0

for i in range(1000):
    start = timeutil.get_epochtime_ms()
    x1, x2, y1, y2, box_h = detection.detect(IMG_PATH='sample/resize_2.jpg', conf_thres=0.5, nms_thres=0.5)  # output
    total_time += timeutil.get_epochtime_ms() - start
    # print("Latency: %fms" % (timeutil.get_epochtime_ms() - start))

print('avg time is '+str(total_time/1000))