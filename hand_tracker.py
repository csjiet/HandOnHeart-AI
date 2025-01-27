import os
import gc
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.cm as cm
from PIL import Image
import numpy as np
import cv2
import torch
from omegaconf import OmegaConf
import hydra
from pprint import pprint
import enum 
import pandas as pd

sys.path.append('./mediapipe')
import mediapipe as mp
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision
# 
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec

repo_path = os.path.abspath("./sam2")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor

from scipy.stats import gaussian_kde

import subprocess

def slice_video_to_frames(video_path, output_dir):
    # Check if the video file exists
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' not found.")
    
    # Check if the op directory exists
    if not os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' does not exist. Creating it.")
        os.makedirs(output_dir)
    else:
        print(f"Output directory '{output_dir}' already exists. Skipping slicing.")
        return

    ffmpeg_command = [
        "ffmpeg",
        "-i", video_path,  # Input video path
        "-q:v", "2",       # High-quality frames
        "-start_number", "0",  # Start numbering frames from 0
        os.path.join(output_dir, "%05d.jpg")  # Output frame format
    ]

    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"Frames saved to '{output_dir}' successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during slicing video: {e}")

def show_mask(mask, ax, obj_id=None, random_color=False): 
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(patches.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

predictor = None 
inference_state = None

def save_segmented_video(video_segments, frame_names, video_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for frame_idx in range(len(frame_names)):
        frame_path = os.path.join(video_dir, frame_names[frame_idx])
        frame = cv2.imread(frame_path)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame_idx in video_segments:
            for obj_id, mask in video_segments[frame_idx].items():
                # Ensure mask is reshaped to match frame dimensions
                mask = np.squeeze(mask)  
                if mask.shape[:2] != frame.shape[:2]:
                    raise ValueError(
                        f"Mask shape {mask.shape} does not match frame shape {frame.shape[:2]}"
                    )
                
                # Generate colored mask
                colored_mask = np.zeros_like(frame, dtype=np.uint8)
                # color = np.random.randint(50, 255, size=(3,), dtype=np.uint8)
                # color = (0, 165, 255)  # BGR for a medium orange
                # color = (255, 165, 0)  # BGR for a medium orange
                color = (200, 100, 0)  # BGR for a medium orange
                colored_mask[mask > 0] = color  # Apply mask to color

                # Blend the frame with the colored mask
                # frame = cv2.addWeighted(frame, 0.5, colored_mask, 0.8, 0)
                blended_region = cv2.addWeighted(frame, 0.5, colored_mask, 0.8, 0)
                frame[mask > 0] = blended_region[mask > 0]  # Replace only the segmented pixels

        # Save the processed frame
        output_frame_path = os.path.join(output_dir, f"{frame_idx:05d}.jpg")
        cv2.imwrite(output_frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Combine frames into a video
    output_video_path = os.path.join(output_dir, "segmented_video.mp4")
    frame_rate = 30  # Adjust as needed
    first_frame = cv2.imread(os.path.join(output_dir, f"{0:05d}.jpg"))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    for frame_idx in range(len(frame_names)):
        frame_path = os.path.join(output_dir, f"{frame_idx:05d}.jpg")
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()
    print(f"Segmented video saved at {output_video_path}!")


def sam2_interface(cartesian_dp, frame_name, video_dir, index, radius=50, grid_spacing=100, save_video=False, output_dir="./output_frames/segmented_non_stroke_frames"):
    global predictor
    global inference_state

    print(f"Save video is set to {save_video}")
    
    if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    if not predictor:
        # Initialize the SAM 2 predictor
        sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"

        config_dir = os.path.abspath("./sam2/sam2/configs")
        config_name = "configs/sam2.1_hiera_l.yaml"

        hydra.initialize_config_dir(config_dir=config_dir, version_base="1.2")

        device = torch.device("cpu")
        predictor = build_sam2_video_predictor(config_name, sam2_checkpoint, device=device)

        predictor = predictor.to(device)


    inference_state = predictor.init_state(video_path=video_dir)

    plt.figure(figsize=(9, 6))
    plt.title(f"frame: {frame_name}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_name)))

    image = cv2.imread(os.path.join(video_dir, frame_name))
    image_height, image_width, _ = image.shape

    # Generate a grid of points
    x_coords = np.arange(0, image_width, grid_spacing)
    y_coords = np.arange(0, image_height, grid_spacing)
    grid_points = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)

    # Calculate distances from each grid point to all positive points
    hand_points = np.array(cartesian_dp)  # Positive points from Mediapipe landmarks
    distances = np.linalg.norm(grid_points[:, None, :] - hand_points[None, :, :], axis=2)

    # Filter grid points based on the distance threshold
    is_negative = np.all(distances > radius, axis=1)  # Points farther than `radius` from all hand points
    negative_points = grid_points[is_negative]

    # Combine positive and negative points
    points = np.vstack((hand_points, negative_points))
    labels = np.array([1] * len(hand_points) + [0] * len(negative_points), dtype=np.int32)

    # # Add a positive click at (x, y) 
    # points = np.array(cartesian_dp, dtype=np.float32)
    # # for labels, `1` means positive click and `0` means negative click
    # labels = np.array(np.ones(len(cartesian_dp)), np.int32)

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx= index,
        # obj_id= index + 1,
        obj_id = 1,
        points=points,
        labels=labels,
    )

    # show the results on the current (interacted) frame
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {frame_name}")

    plt.imshow(Image.open(os.path.join(video_dir, frame_name)))
    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

    plt.show()

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # render the segmentation results every few frames
    vis_frame_stride = 50
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        if out_frame_idx not in video_segments:
            print(f"skipping frame {out_frame_idx} due to missing video_segments :(.")
            continue
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    print("Saving video...")
    if save_video:
        save_segmented_video(video_segments, frame_names, video_dir, output_dir)




class CopyHandLandmark(enum.IntEnum):
  """The 21 hand landmarks."""

  WRIST = 0
  THUMB_CMC = 1
  THUMB_MCP = 2
  THUMB_IP = 3
  THUMB_TIP = 4
  INDEX_FINGER_MCP = 5
  INDEX_FINGER_PIP = 6
  INDEX_FINGER_DIP = 7
  INDEX_FINGER_TIP = 8
  MIDDLE_FINGER_MCP = 9
  MIDDLE_FINGER_PIP = 10
  MIDDLE_FINGER_DIP = 11
  MIDDLE_FINGER_TIP = 12
  RING_FINGER_MCP = 13
  RING_FINGER_PIP = 14
  RING_FINGER_DIP = 15
  RING_FINGER_TIP = 16
  PINKY_MCP = 17
  PINKY_PIP = 18
  PINKY_DIP = 19
  PINKY_TIP = 20

midknuckle = CopyHandLandmark.MIDDLE_FINGER_MCP

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

def initialize_holistic_model():
    holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode= False
    )
    return holistic_model

def initialize_hand_model():
    holistic_model = mp_hands.Hands(
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1,
    static_image_mode= False
    )
    return holistic_model

def mediapipe_inference(VIDEO_DIR, file, selected_landmarks):
    # Initialize MediaPipe Hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Load the image
    if os.path.isabs(file):  # If `file` is an absolute path, use it directly
        image_path = file
    else:  # Otherwise, combine `VIDEO_DIR` and `file`
        image_path = os.path.join(VIDEO_DIR, file)

    print(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image could not be loaded. Check the file path: {image_path}")

    image_height, image_width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize the Hands model
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3
    )

    # Process the image
    results = hands.process(image_rgb)
    annotated_image = image.copy()

    # Draw hand landmarks if detected
    left_hand_landmarks = []
    right_hand_landmarks = []
    all_hand_landmarks = []  # To store all hand landmark coordinates
    row_data = {}  # To store a row of data for the current frame

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Determine hand label (right or left)
            hand_label = "rh" if results.multi_handedness[hand_idx].classification[0].label == "Right" else "lh"

            # Draw landmarks on the image
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            )

            # Extract selected landmarks
            for idx, landmark in enumerate(hand_landmarks.landmark):
                if idx in selected_landmarks and selected_landmarks[idx]:
                    x_pixel = int(landmark.x * image_width)
                    y_pixel = int(landmark.y * image_height)

                    landmark_name = CopyHandLandmark(idx).name
                    col_name = f"{hand_label}_{landmark_name}"  # Prefix with hand label
                    row_data[col_name] = (x_pixel, y_pixel)

                    if hand_label == "rh":
                        right_hand_landmarks.append([x_pixel, y_pixel])
                    else:
                        left_hand_landmarks.append([x_pixel, y_pixel])

    all_hand_landmarks = left_hand_landmarks + right_hand_landmarks

    # Release the model resources
    hands.close()

    return all_hand_landmarks, annotated_image, row_data


def mediapipe_inference2(VIDEO_DIR, file, selected_landmarks):
    holistic_model = initialize_holistic_model()

    if os.path.isabs(file):  # If `file` is an absolute path, use it directly
        image_path = file
    else:  # Otherwise, combine `VIDEO_DIR` and `file`
        image_path = os.path.join(VIDEO_DIR, file)
    
    # Debugging: Print the image path
    print(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image could not be loaded. Check the file path: {image_path}")
    image_height, image_width, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (640, 640))  # Standard size

    image_height, image_width, _ = image.shape

    # target_size = 640
    # old_size = image.shape[:2]  # (height, width)
    # ratio = float(target_size) / max(old_size)
    # new_size = tuple([int(x * ratio) for x in old_size])

    # image = cv2.resize(image, (new_size[1], new_size[0]))
    # delta_w = target_size - new_size[1]
    # delta_h = target_size - new_size[0]
    # top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    # left, right = delta_w // 2, delta_w - (delta_w // 2)

    # image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


    results = holistic_model.process(image)
    annotated_image = image.copy()

    mp_drawing.draw_landmarks(
      annotated_image, 
      results.right_hand_landmarks, 
      mp_holistic.HAND_CONNECTIONS
    )

    mp_drawing.draw_landmarks(
      annotated_image, 
      results.left_hand_landmarks, 
      mp_holistic.HAND_CONNECTIONS
    )
    
    left_hand_landmarks = []
    right_hand_landmarks = []
    all_hand_landmarks = []  # To store all hand landmark coordinates
    row_data = {}  # To store a row of data for the current frame

    for hand_label, hand_landmarks in zip(["rh", "lh"], [results.right_hand_landmarks, results.left_hand_landmarks]):
        if hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                if idx in selected_landmarks and selected_landmarks[idx]:
                    # Convert normalized coordinates to pixel coordinates
                    x_pixel = int(landmark.x * image_width)
                    y_pixel = int(landmark.y * image_height)

                    landmark_name = CopyHandLandmark(idx).name
                    col_name = f"{hand_label}_{landmark_name}"  # Prefix with hand label
                    row_data[col_name] = (x_pixel, y_pixel)

                    #print(hand_landmarks)
                    #all_hand_landmarks.append([x_pixel, y_pixel])
                    if(hand_label == "rh"):
                        right_hand_landmarks.append([x_pixel, y_pixel])
                    else:
                        left_hand_landmarks.append([x_pixel, y_pixel])
                        
    all_hand_landmarks = left_hand_landmarks + right_hand_landmarks
    return all_hand_landmarks, annotated_image, row_data

def hand_tracker_main_interface(VIDEO_DIR, file, selected_landmarks, index, output_dir):
    all_hand_landmarks, annotated_image, row_data = mediapipe_inference(VIDEO_DIR, file, selected_landmarks)
    if all_hand_landmarks:
        sam2_interface(all_hand_landmarks, file, VIDEO_DIR, index, save_video=True, output_dir= output_dir)

    plt.figure(figsize=(9, 6))
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    plt.title(f'landmarks: {file}')


def extract_hand_features(video_dir, selected_landmarks, output_csv="./hand_features.csv", batch_size=100):
    holistic_model = None

    try:
        # Initialize Mediapipe Holistic model
        holistic_model = initialize_holistic_model()

        # Get all frame filenames
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        # Prepare a list to store features temporarily
        hand_features = []

        for batch_start in range(0, len(frame_names), batch_size):
            batch_frames = frame_names[batch_start:batch_start + batch_size]

            for frame_idx, file in enumerate(batch_frames):
                print(f"Processing frame {batch_start + frame_idx + 1}/{len(frame_names)}: {file}")

                # Extract features for the current frame
                _, _, row_data = mediapipe_inference(video_dir, file, selected_landmarks)
                row_data["frame"] = file  # Add frame identifier
                hand_features.append(row_data)

            # Save batch data to CSV
            if hand_features:
                pd.DataFrame(hand_features).to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
                hand_features.clear()  # Clear memory after writing

                # Force garbage collection
                gc.collect()

        print(f"Hand features successfully saved to {output_csv}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Ensure the model is closed properly
        if holistic_model:
            holistic_model.close()

def heatmap_kde(df, image_path="./video_frames/non_stroke_frames/00000.jpg"):
    # Load and clean data
    # df = pd.read_csv(file_path)
    df = df.dropna()

    def converter_func1(val):
        if isinstance(val, str):  # Check if the value is a string
            coord_tuple = tuple(map(int, val.strip("()").split(",")))
            return np.array(coord_tuple)

    all_coordinates = []
    for col in df.columns:
        if col != "frame":
            df[col] = df[col].apply(converter_func1)  # Apply conversion function
            all_coordinates.extend(df[col].dropna().values)

    coordinates_2d = np.array(all_coordinates)

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} could not be loaded.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image_height, image_width, _ = image.shape

    coord_2d = coordinates_2d.T
    x_coords, y_coords = coord_2d[0], coord_2d[1]

    # 2D KDE
    kde = gaussian_kde(coord_2d)
    xmin, xmax = x_coords.min(), x_coords.max()
    ymin, ymax = y_coords.min(), y_coords.max()

    # Create grid for KDE
    x_grid, y_grid = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z_grid = kde(positions).reshape(x_grid.shape)

    # 3D KDE Plot
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    z_scale = 1
    z_grid_scaled = z_grid * z_scale

    ax.plot_surface(
        x_grid,
        y_grid,
        z_grid_scaled,
        cmap="PuOr",
        edgecolor="none",
        alpha=0.6,
        antialiased=True,
    )

    image_array = cv2.resize(image, (80, 80)) / 255.0
    x_img = np.linspace(xmin, xmax, 80)
    y_img = np.linspace(ymin, ymax, 80)
    x_img, y_img = np.meshgrid(x_img, y_img)

    ax.plot_surface(
        x_img,
        y_img,
        np.zeros_like(x_img),  # z=0
        rstride=1,
        cstride=1,
        facecolors=image_array,
        shade=False,
        alpha=0.8,
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(z_grid.min(), z_grid.max())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("kernel density estimate")
    ax.set_title("KDE heatmap (3D)")

    # 2D KDE Plot
    fig, ax = plt.subplots(figsize=(9, 7))
    flipped_image = cv2.flip(image, 0)  # Flip vertically for proper alignment
    ax.imshow(
        flipped_image,
        extent=(xmin, xmax, ymin, ymax),
        origin="lower",
        alpha=0.8,
    )
    heatmap = ax.imshow(
        z_grid,
        extent=(xmin, xmax, ymin, ymax),
        origin="lower",
        cmap="PuOr",
        alpha=0.7,
    )
    fig.colorbar(heatmap, ax=ax, label="Density")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("KDE heatmap (2D)")

    plt.show()

def plot_hand_motion(df, image_path="./video_frames/non_stroke_frames/00000.jpg"):
    # Clean and validate data
    df = df.dropna()

    def converter_func1(val):
        if isinstance(val, str): 
            coord_tuple = tuple(map(int, val.strip("()").split(",")))
            return np.array(coord_tuple)
    
    for col in df.columns:
        if col != "frame":
            df[col] = df[col].apply(converter_func1)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} could not be loaded.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = image.shape

    # Prepare the plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Motion tracking across frames")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Display the image in the background
    ax.imshow(image, extent=(0, image_width, image_height, 0), origin="upper")

    landmark_columns = [col for col in df.columns if col != "frame"]
    colors = cm.get_cmap('tab20', len(landmark_columns))

    for i, col in enumerate(landmark_columns):
        landmark_coords = np.array(df[col].tolist())
        x_coords = landmark_coords[:, 0]
        y_coords = landmark_coords[:, 1]

        ax.plot(x_coords, y_coords, label=col, color=colors(i), linewidth=1.5)
        # Mark first point with a unique marker
        ax.scatter(
            x_coords[0], y_coords[0], 
            color=colors(i), 
            edgecolor="black", 
            s=100, 
            marker="o", 
        )

    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)
    ax.legend(
        title="Landmarks", 
        loc="upper center", 
        bbox_to_anchor=(0.5, -0.12),  # Push the legend below the plot
        fontsize="x-small", 
        title_fontsize="small", 
        ncol=5,  # Number of columns in the legend
        frameon=False  # Remove legend border
    )

    # Add padding to prevent overlap of legend and x-label
    plt.subplots_adjust(bottom=0.3)  # Increase bottom margin to make space for legend
    plt.tight_layout()
    plt.show()
