import os
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh


def extract_all_videos(speaker_dir, output_root):
    # Modified paths to match your structure
    video_dir = os.path.join(speaker_dir, "s1_videos")  # Changed from "video" to "s1_videos"
    align_dir = os.path.join(speaker_dir, "align")
    os.makedirs(output_root, exist_ok=True)

    for video_file in os.listdir(video_dir):
        if not video_file.endswith(".mpg"):
            continue

        video_path = os.path.join(video_dir, video_file)
        base_name = video_file[:-4]  # Remove .mpg
        output_dir = os.path.join(output_root, base_name)
        os.makedirs(output_dir, exist_ok=True)

        extract_mouth_from_video(video_path, output_dir)

        # Optionally, save transcript
        align_path = os.path.join(align_dir, base_name + ".align")
        if os.path.exists(align_path):
            with open(align_path, 'r') as f:
                words = [line.split()[2] for line in f if "sil" not in line]
                with open(os.path.join(output_dir, "transcript.txt"), "w") as out:
                    out.write(" ".join(words))


def extract_mouth_from_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    with mp_face_mesh.FaceMesh(static_image_mode=False) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    mouth_pts = [landmarks.landmark[i] for i in range(61, 88)]
                    x = [int(p.x * w) for p in mouth_pts]
                    y = [int(p.y * h) for p in mouth_pts]
                    x_min, x_max = max(min(x) - 10, 0), min(max(x) + 10, w)
                    y_min, y_max = max(min(y) - 10, 0), min(max(y) + 10, h)

                    mouth_crop = frame[y_min:y_max, x_min:x_max]
                    mouth_crop = cv2.resize(mouth_crop, (112, 112))
                    cv2.imwrite(f"{output_dir}/frame_{frame_idx:04d}.png", mouth_crop)
                    break  # Only use first detected face

            frame_idx += 1
    cap.release()


if __name__ == "__main__":
    # Using your exact paths
    speaker_dir = "/Users/rishabhbhargav/PycharmProjects/LipReadingAI/data/s1"
    output_root = "/Users/rishabhbhargav/PycharmProjects/LipReadingAI/data/s1/mouth_crops"  # Output directory

    extract_all_videos(speaker_dir, output_root)