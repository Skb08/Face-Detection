import os
import argparse

import cv2
import mediapipe as mp


def process_img(img, face_detection):
    if img is None:
        print("Error: Failed to read the image.")
        return None

    H, W, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # print(x1, y1, w, h)

            # blur faces
            # img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (30, 30))

            #rectangle faces
            img = cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,200,0),25)

    return img


def main():
    args = argparse.ArgumentParser()

    #for image
    args.add_argument("--mode", default='image')
    args.add_argument("--filePath", default='./img1.png')

    # # for video
    # args.add_argument("--mode", default='video')
    # args.add_argument("--filePath", default='./video1.mp4')

    args = args.parse_args()

    output_dir = './output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # detect faces
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        if args.mode in ["image"]:
            # Print file path for debugging
            print("File path:", args.filePath)

            # read image
            img = cv2.imread(args.filePath)

            img = process_img(img, face_detection)

            if img is not None:
                # save image
                cv2.imwrite(os.path.join(output_dir, 'output.png'), img)

        elif args.mode in ['video']:
            cap = cv2.VideoCapture(args.filePath)
            ret, frame = cap.read()

            output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),
                                           cv2.VideoWriter_fourcc(*'MP4V'),
                                           25,
                                           (frame.shape[1], frame.shape[0]))

            while ret:
                frame = process_img(frame, face_detection)

                if frame is not None:
                    output_video.write(frame)

                ret, frame = cap.read()

            cap.release()
            output_video.release()

        elif args.mode in ['webcam']:
            cap = cv2.VideoCapture(0)

            ret, frame = cap.read()
            while ret:
                frame = process_img(frame, face_detection)

                if frame is not None:
                    cv2.imshow('frame', frame)
                    cv2.waitKey(25)

                ret, frame = cap.read()

            cap.release()


if __name__ == "__main__":
    main()
