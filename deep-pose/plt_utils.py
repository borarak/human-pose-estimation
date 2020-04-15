from matplotlib import pyplot as plt
import io
import cv2
import numpy as np
import tensorflow as tf

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255,
                                                     0], [170, 255, 0],
          [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255],
          [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0,
                                                    255], [255, 0, 255],
          [125, 0, 125], [0, 255, 255], [170, 255, 0]]

pairs = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
         [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
         [2, 4], [3, 5], [4, 6], [5, 7]]


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def get_joints(res, threshold=0.85):
    final_kp = [
        tf.where((res[:, :, idx] == tf.reduce_max(res[:, :, idx]))
                 & (res[:, :, idx] >= threshold)) for idx in range(17)
    ]
    return final_kp


def plot_results(input_images, predicted_keypoints, save_fig=False):
    input_images = input_images.numpy()
    predicted_keypoints = predicted_keypoints.numpy()

    plt.rcParams['figure.figsize'] = (15, 10)
    if input_images.shape[0] <= 1:
        fig, ax = plt.subplots(1, 3)
        max_plots = 1
    else:
        max_plots = min(4, input_images.shape[0])
        fig, ax = plt.subplots(max_plots, 3)

    for idx, image in enumerate(input_images):
        if idx >= max_plots:
            break
        # t_kp = np.transpose(
        #     np.sum(target_keypoints[idx], axis=-1, keepdims=False))
        # t_kp = np.stack([t_kp, t_kp, t_kp], axis=-1)

        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        image = cv2.resize(image, (64, 64))
        image_cp = image.copy()

        # print(predicted_keypoints[idx].shape)
        p_kp = np.transpose(np.max(predicted_keypoints[idx][:, :, :], axis=-1))
        p_kp = np.stack([p_kp, p_kp, p_kp], axis=-1)
        p_kp = np.clip(p_kp * 255, 0, 255).astype(np.uint8)

        blend = cv2.addWeighted(image, 0.5, p_kp, 0.5, 0.0)
        blend = cv2.cvtColor(cv2.resize(blend, (512, 512)), cv2.COLOR_RGB2BGR)

        p_kp_up = cv2.resize(p_kp, (512, 512))
        joints_img = plot_joints(image_cp, predicted_keypoints[idx], save_fig)

        if len(ax.shape) <= 1:
            ax[0].imshow(p_kp_up)
            ax[1].imshow(blend)
            ax[2].imshow(joints_img)
        else:
            ax[idx, 0].imshow(p_kp_up)
            ax[idx, 1].imshow(blend)
            ax[idx, 2].imshow(joints_img)

    return plot_to_image(fig)


def plot_results2(input_images,
                  predicted_keypoints,
                  save_fig=False,
                  scale=1.0):
    input_images = input_images
    predicted_keypoints = predicted_keypoints.numpy()

    plt.rcParams['figure.figsize'] = (15, 10)
    if input_images.shape[0] <= 1:
        fig, ax = plt.subplots(1, 3)
        max_plots = 1
    else:
        max_plots = min(4, input_images.shape[0])
        fig, ax = plt.subplots(max_plots, 3)

    for idx, image in enumerate(input_images):
        if idx >= max_plots:
            break
        # t_kp = np.transpose(
        #     np.sum(target_keypoints[idx], axis=-1, keepdims=False))
        # t_kp = np.stack([t_kp, t_kp, t_kp], axis=-1)

        # image = np.clip(image * 255, 0, 255).astype(np.uint8)
        # image = cv2.resize(image, (64, 64))
        image_cp = image.copy()

        # print(predicted_keypoints[idx].shape)
        p_kp = np.transpose(np.max(predicted_keypoints[idx][:, :, :], axis=-1))
        p_kp = np.stack([p_kp, p_kp, p_kp], axis=-1)
        p_kp = np.clip(p_kp * 255, 0, 255).astype(np.uint8)

        p_kp = cv2.resize(p_kp, (256, 256))
        blend = cv2.addWeighted(image, 0.7, p_kp, 0.3, 0.0)
        blend = cv2.cvtColor(cv2.resize(blend, (512, 512)), cv2.COLOR_RGB2BGR)

        p_kp_up = cv2.resize(p_kp, (512, 512))
        joints_img = plot_joints(image_cp, predicted_keypoints[idx], save_fig,
                                 scale)

        if len(ax.shape) <= 1:
            ax[0].imshow(p_kp_up)
            ax[1].imshow(blend)
            ax[2].imshow(joints_img)
        else:
            ax[idx, 0].imshow(p_kp_up)
            ax[idx, 1].imshow(blend)
            ax[idx, 2].imshow(joints_img)

    return plot_to_image(fig)


def plot_joints(input_images, predicted_keypoints, save_fig, scale=1.0):
    image = input_images
    joints = get_joints(predicted_keypoints)

    # print("joints: ", joints)

    points = []
    for idx, kp in enumerate(joints):
        kp = kp.numpy()
        if len(kp) <= 0:
            points.append(None)
            continue

        kp = kp[0]  # Taking the first of all equal valued joints
        #        _kp = list(reversed(list(map(int, kp * 256 / 64))))
        _kp = list(map(int, kp * scale))
        cv2.circle(image, tuple(_kp), radius=1, color=colors[idx], thickness=1)
        points.append(tuple(_kp))

    # print("points: ", points)
    for idx, pair in enumerate(pairs):
        if pair is None:
            continue
        p1 = points[pair[0] - 1]
        p2 = points[pair[1] - 1]

        if p1 is None or p2 is None:
            continue
        # print(f"p1: {p1}, p2: {p2}")
        cv2.line(image, p1, p2, colors[idx], 1)

    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if save_fig:
        cv2.imwrite("~/workspace/human-pose-estimation/data/test_result.jpg",
                    image)
    return image
