import numpy as np


def estimate_depth():
    pass


def get_camera_info_file_path(frame_seq=0, base_dir='dataset/fountain-P11'):
    camera_info_dir = 'gt_dense_cameras'
    camera_info_file_path = "{}/{}/{:04d}.jpg.camera".format(
        base_dir, camera_info_dir, frame_seq)
    return camera_info_file_path


def get_camera_info(frame_seq=0):
    info_path = get_camera_info_file_path(frame_seq)
    info_file = open(info_path, 'r')
    info = info_file.read().split('\n')
    K = np.reshape(np.fromstring(' '.join(info[0:3]), sep=' '), (3, 3))
    R = np.reshape(np.fromstring(' '.join(info[4:4+3]), sep=' '), (3, 3))
    C = np.reshape(np.fromstring(
        ' '.join(info[7:8]), sep=' '), (1, 3)).flatten()
    dims = np.reshape(np.fromstring(
        ' '.join(info[8:9]), sep=' '), (2, 1)).flatten()
    t = np.matmul(R.transpose(), C)
    assert(np.abs(np.matmul(R, R.transpose()) - np.eye(3)).sum() < 1e-3)
    print('K\n', K, '\nR\n', R, '\nt\n', t)
    T_f_w = np.eye(4)
    T_f_w[:3, :3] = R
    T_f_w[:3, 3] = t
    return K, T_f_w, dims


def homogenize(x):
    assert(len(x.shape) == 1)
    return np.concatenate([x, [1]])


def dehomogenize(x_dot):
    return x_dot[:-1]


def project_pinhole(x):
    return dehomogenize(x)/x[-1]


'''
 plot epipolar line in image b corresponding to point p in image a
 pose_b_a  is se3 transform form A to B
'''


def plot_epipolar_line(p_a, pose_b_a, K):
    z_p_a = 1
    # eqn. 3.11 [1]
    p_dot = np.array([p_a[0], p_a[1], 1])
    p_w = z_p_a * np.matmul(np.linalg.inv(K), p_dot)
    print(p_w)
    p_b = project_pinhole(np.matmul(K, dehomogenize(
        np.matmul(pose_b_a, homogenize(p_w)))))
    print('p_b', p_b)


if __name__ == "__main__":
    np.set_printoptions(precision=3)

    seq_len = 11
    a_idx = seq_len//2
    b_idx = 0
    K, T_a_w, img_dim = get_camera_info(a_idx)
    _, T_b_w, _ = get_camera_info(b_idx)

    p_a = img_dim/2
    T_b_a = T_b_w * np.linalg.inv(T_a_w)
    plot_epipolar_line(p_a, T_b_a, K)
