import numpy as np
from scipy.spatial.transform import Rotation as Rot
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def get_image_file_path(frame_seq=0, base_dir='dataset/fountain-P11'):
    images_dir = 'images'
    seq_image_file_path = "{}/{}/{:04d}.jpg".format(
        base_dir, images_dir, frame_seq)
    return seq_image_file_path


def read_image(image_path=get_image_file_path()):
    return cv2.imread(image_path)


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # draw circle here (etc...)
        print('x = {}, y = {}'.format(x, y))


def display_image(img=read_image()):
    cv2.imshow('image', img)
    # cv2.setMouseCallback('image', onMouse)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_camera_info_file_path(frame_seq=0, base_dir='dataset/fountain-P11'):
    camera_info_dir = 'gt_dense_cameras'
    camera_info_file_path = "{}/{}/{:04d}.jpg.camera".format(
        base_dir, camera_info_dir, frame_seq)
    return camera_info_file_path


def compose_se3(R, t):
    T_f_w = np.eye(4)
    T_f_w[:3, :3] = R
    T_f_w[:3, 3] = t
    return T_f_w


def decompose_se3(T):
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t


def debug_se3(T):
    R, t = decompose_se3(T)
    R_axis = Rot.from_matrix(R).as_rotvec()
    angle = np.linalg.norm(R_axis)
    axis = R_axis/angle
    # print('axis:\t', axis, '\tangle:\t', np.rad2deg(angle), '\tt:\t', t)
    print('ZXY:\t', Rot.from_matrix(R).as_euler(
        'zxy', degrees=True), '\tt:\t', t)


def get_camera_info(frame_seq=0):
    info_path = get_camera_info_file_path(frame_seq)
    info_file = open(info_path, 'r')
    info = info_file.read().split('\n')
    K = np.reshape(np.fromstring(' '.join(info[0:3]), sep=' '), (3, 3))
    # Rotation from camera frame to worldframe
    R_w_f = np.reshape(np.fromstring(' '.join(info[4:4+3]), sep=' '), (3, 3))
    # translation from camera origin to world origin after rotation
    t_fo_wo = np.reshape(np.fromstring(
        ' '.join(info[7:8]), sep=' '), (1, 3)).flatten()
    T_w_f = compose_se3(R_w_f, t_fo_wo)
    T_f_w = np.linalg.inv(T_w_f)
    dims = np.reshape(np.fromstring(
        ' '.join(info[8:9]), sep=' '), (2, 1)).flatten()
    assert(np.abs(np.matmul(R_w_f, R_w_f.transpose()) - np.eye(3)).sum() < 1e-3)
    # print('K\n', K, '\nR\n', R, '\nt\n', t)
    return K, T_f_w, dims


def homogenize(x):
    assert(len(x.shape) == 1)
    return np.concatenate([x, [1]])


def dehomogenize(x_dot):
    assert(np.abs(x_dot[-1]-1) < 1e-3)
    return x_dot[:-1]


def project_pinhole(x):
    return x/x[-1]


def plot_stereo_correspondence(p_a, p_b, img_a, img_b):
    img_a = cv2.circle(img_a, tuple(p_a), 3, color=(0, 255, 0), thickness=-1)
    img_b = cv2.circle(img_b, tuple(p_b), 3, color=(0, 255, 0), thickness=-1)

    fig, ax = plt.subplots(2)
    fig.suptitle('Stereo Correspondence')
    ax[0].imshow(img_a)
    ax[1].imshow(img_b)
    plt.show()


def get_point_correspondence(p_a, z_p_a, T_b_a, K):
    p2_dot_a = homogenize(p_a)
    # eqn. 3.11 [1]
    p3_a = z_p_a * np.matmul(np.linalg.inv(K), p2_dot_a)
    p_dot_b = np.matmul(T_b_a, homogenize(p3_a))
    p2_b = dehomogenize(
        np.matmul(K, project_pinhole(dehomogenize(p_dot_b))))
    return np.round(p2_b)


def plot_epipolar_line(p_a, z_p_a_est, T_b_a, K, b_idx):
    '''plot epipolar line in image b corresponding to point p in image a
    pose_b_a  is se3 transform form A to B'''
    p_b = []
    depth_eps = 0.5
    p_b.append(get_point_correspondence(p_a, z_p_a_est-depth_eps, T_b_a, K))
    p_b.append(get_point_correspondence(p_a, z_p_a_est+depth_eps, T_b_a, K))
    p_b = list(map(lambda p: tuple(p.astype(int)), p_b))

    # plotting
    img = read_image(get_image_file_path(b_idx))
    img = cv2.line(img=img, pt1=p_b[0],
                   pt2=p_b[1], color=(0, 255, 0), thickness=2)
    display_image(img)


def estimate_depth(p_a, p_b_gt, T_b_a, K):
    '''given point correspondences in 2 frames, estimate depth of that point'''
    for z_p_a in np.linspace(6, 8, num=10):
        p2_b = get_point_correspondence(p_a, z_p_a, T_b_a, K)
        print('depth: {:.1f}, error: {:.1f}'.format(
              z_p_a, np.linalg.norm(p2_b - p_b_gt)))


def get_photometric_loss(px_a, px_b):
    assert px_a.shape == px_b.shape and len(px_a.shape) == 1
    return np.linalg.norm(px_b-px_a)


def plot_photometric_loss(p_a, z_p_a_est, T_b_a, K, img_a, img_b, do_plot=False):
    '''Computes photometric loss vs depth along epipolar line'''
    dims = np.flip(img_a.shape[:-1])
    depth_eps = 0.5
    depths = []
    p_loss = []
    for z_p_a in np.linspace(z_p_a_est-depth_eps, z_p_a_est+depth_eps, 100):
        p_b = get_point_correspondence(
            p_a, z_p_a, T_b_a, K)
        p_b = p_b.astype(int)
        if np.all(p_b < dims):
            depths.append(z_p_a)
            p_loss.append(get_photometric_loss(
                img_a[tuple(np.flip(p_a))], img_b[tuple(np.flip(p_b))]))
            # plot_stereo_correspondence(p_a, p_b, img_a, img_b)

    depths = np.array(depths)
    p_loss = np.array(p_loss)
    if do_plot:
        fig, ax = plt.subplots()
        ax.plot(depths, p_loss)
        ax.set(xlabel='depth', ylabel='photometric loss',
               title='Loss along epipolar line')
        plt.show()
    return depths, p_loss


def add_smooth_plot(x, y, label, thick_line=False):
    y_spline = make_interp_spline(x, y)
    x_fine = np.linspace(x[0], x[-1], 1000)
    y_fine = y_spline(x_fine)
    thickness = 1.0
    if thick_line:
        thickness = 3.0
    plt.plot(x_fine, y_fine, label=label, linewidth=thickness)


if __name__ == "__main__":
    np.set_printoptions(precision=3)

    seq_len = 11
    a_idx = 0
    # bottom right cross marker
    p_a = np.array([2319, 1866])
    depth_a = 6.9
    # p_b = np.array([2810, 1647])
    K, T_a_w, img_dim = get_camera_info(a_idx)

    seq_depths = []
    seq_p_loss = []

    img_a = read_image(get_image_file_path(a_idx))

    fig = plt.figure(figsize=(14, 8))
    plt.title(
        'Photometric loss for right-bottom marker point in fountain-P11 sequence')
    plt.xlabel('depth')
    plt.ylabel('photometric loss')

    for b_idx in range(1, seq_len):
        img_b = read_image(get_image_file_path(b_idx))
        _, T_b_w, _ = get_camera_info(b_idx)
        T_b_a = np.matmul(T_b_w, np.linalg.inv(T_a_w))
        depth, p_loss = plot_photometric_loss(p_a, depth_a, T_b_a, K,
                                              img_a=img_a, img_b=img_b)
        seq_depths.append(depth)
        seq_p_loss.append(p_loss)
        add_smooth_plot(depth, p_loss, 'seq {}'.format(b_idx))

    add_smooth_plot(seq_depths[0], sum(seq_p_loss) /
                    len(seq_p_loss), 'seq avg.', True)
    plt.legend()
    plt.savefig('photometric_loss_vs_depth__Fountain_P11.png')
    plt.show()

    # plot_stereo_correspondence(a_gt, b_gt, img_a, img_b)
    # estimate_depth(a_gt, b_gt, T_b_a, K)
    # p_b = plot_epipolar_line(p_a, T_b_a, K)
