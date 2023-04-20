import numpy as np
import cv2


def YOLO(image_addres, model_addres, label_addres):

    # step 1 - load the model

    net = cv2.dnn.readNet(model_addres)

    # step 2 - feed a 640x640 image to get predictions

    def format_yolov5(frame):

        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

    image = cv2.imread(image_addres)
    input_image = format_yolov5(image)  # making the image square
    blob = cv2.dnn.blobFromImage(input_image, 1/255.0, (640, 640), swapRB=True)
    net.setInput(blob)
    predictions = net.forward()

    # step 3 - unwrap the predictions to get the object detections

    class_ids = []
    confidences = []
    boxes = []

    output_data = predictions[0]

    image_width, image_height, _ = input_image.shape
    x_factor = image_width / 640
    y_factor = image_height / 640

    for r in range(25200):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(
                ), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    # Step 4 — Printing the resulting image

    class_list = []
    with open(label_addres, "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]

    colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

    for i in range(len(result_class_ids)):

        box = result_boxes[i]
        class_id = result_class_ids[i]

        color = colors[class_id % len(colors)]

        conf = result_confidences[i]

        cv2.rectangle(image, box, color, 2)
        cv2.rectangle(image, (box[0], box[1] - 20),
                      (box[0] + box[2], box[1]), color, -1)
        cv2.putText(image, class_list[class_id], (box[0] + 5,
                    box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    image_addres_list = list(image_addres)
    while (image_addres_list != 0):
        if image_addres_list.pop() != '/':
            image_addres_list.pop()
        else:
            break
    image_addres_without_name = ''.join(image_addres_list)
    image_addres_with_name = image_addres_without_name + r'/image_inference.jpg'

    cv2.imwrite(image_addres_with_name, image)
