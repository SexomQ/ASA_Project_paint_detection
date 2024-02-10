from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import playsound as ps
import multiprocessing as mp
import time
import threading

# This function is used to play the music
def music_player(play_flag):
    while play_flag.is_set():
        ps.playsound("Music/CountrySounds-short.mp3")

def main():
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("Model/converted_keras/keras_model.h5", compile=False)

    # Load the labels
    class_names = open("Model/converted_keras/labels.txt", "r").readlines()

    # CAMERA can be 0 or 1 based on default camera of your computer
    camera = cv2.VideoCapture(0)

    # Create a process for the music player
    play_flag = threading.Event()
    play_flag.set()
    song_thread = threading.Thread(target=music_player, args=(play_flag,))
    # song_thread.start()

    keep_playing = False
    last_detection_time = time.time()

    while True:
        # Grab the webcamera's image.
        ret, image = camera.read()

        # Resize the raw image into (224-height,224-width) pixels
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # Show the image in a window
        cv2.imshow("Webcam Image", image)

        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Predicts the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

        # If the class name is "Tablou", then play the sound
        if class_name[2:] == class_names[1][2:]:
            if not keep_playing:
                play_flag.set()
                song_thread = threading.Thread(target=music_player, args=(play_flag,))
                song_thread.start()
                last_detection_time = time.time()
                keep_playing = True

        # If the class name is "Empty", then stop the sound
        else:
            if keep_playing and (time.time() - last_detection_time > 4):
                play_flag.clear()
                keep_playing = False
            

        # Listen to the keyboard for pressed.
        keyboard_input = cv2.waitKey(1)

        # 27 is the ASCII for the esc key on your keyboard.
        if keyboard_input == 27:
            play_flag.clear()
            song_thread.join()
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()