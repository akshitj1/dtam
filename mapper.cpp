#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <fmt/format.h>

using namespace std;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

void display_image(const string image_file_path = "dataset/fountain-P11/images/0000.jpg")
{
    cv::Mat image = cv::imread(image_file_path, cv::IMREAD_COLOR);
    assert(image.data);
    cv::namedWindow("print image", cv::WINDOW_AUTOSIZE);
    cv::imshow("print image", image);
    cv::waitKey(0);
}

string get_camera_info_file_path(const int frame_seq, const string base_dir = "dataset/fountain-P11")
{
    const string camera_info_dir = "gt_dense_cameras";
    const string camera_info_file_path = fmt::format("{}/{}/{:04d}.jpg.camera", base_dir, camera_info_dir, frame_seq);
    return camera_info_file_path;
}

string get_image_file_path(const int frame_seq, const string base_dir = "dataset/fountain-P11")
{
    const string images_dir = "images";
    const string image_file_path = fmt::format("{}/{}/{:04d}.jpg", base_dir, images_dir, frame_seq);
    return image_file_path;
}

void read_image(const string image_file_path, MatrixXd &image)
{
    cv::Mat img_color = cv::imread(image_file_path, cv::IMREAD_COLOR);
    cv::Mat img_gray;
    cv::cvtColor(img_color, img_gray, cv::COLOR_BGR2GRAY);
    cv::cv2eigen(img_gray, image);
}

void read_camera_info(const string camera_info_file_path, Matrix3d &_K, Eigen::Matrix4d &_T)
{
    // https://github.com/openMVG/openMVG/blob/160643be515007580086650f2ae7f1a42d32e9fb/src/software/SfM/import/io_readGTStrecha.hpp#L62
    ifstream camera_info_file(camera_info_file_path, ios::in);
    assert(camera_info_file);
    vector<double> camera_info;
    while (camera_info_file)
    {
        double info_element;
        camera_info_file >> info_element;
        camera_info.push_back(info_element);
    }
    Eigen::Map<Eigen::VectorXd> info(camera_info.data(), camera_info.size());
    Eigen::Map<Eigen::MatrixXd> K(info.block(0, 0, 9, 1).data(), 3, 3);
    K.transposeInPlace();
    Eigen::Map<Eigen::MatrixXd> R(info.block(12, 0, 9, 1).data(), 3, 3);
    Eigen::Map<Eigen::Vector3d> C(info.block(21, 0, 3, 1).data());

    _K = K;

    // Strecha model is P = K[R^T|-R^T t];
    // My model is P = K[R|t], t = - RC
    Eigen::VectorXd t = R.transpose() * C;
    R.transposeInPlace();

    assert((R * R.transpose()).isIdentity(1e-4));
    _T = Eigen::Matrix4d::Identity();
    _T.block(0, 0, 3, 3) = R;
    _T.block(0, 3, 3, 1) = t;
}

Eigen::Vector2i project_pinhole(Vector3d p)
{
    Eigen::Vector2d p_img_d = p.block(0, 0, 2, 1) / p.z();
    Eigen::Vector2i p_img = p_img_d.cast<int>(); //.array().round();
    return p_img;
}

Eigen::Vector4d homonogenize(const Vector3d &x)
{
    Eigen::Vector4d x_dot(x.x(), x.y(), x.z(), 1);
    return x_dot;
}

Eigen::Vector3d dehomonogenize(const Eigen::Vector4d &x)
{
    return x.block(0, 0, 3, 1);
}

/**
 * Calculates photometric loss given pixel u in ref. image A w.r.t image B
 */
void photometric_loss(
    const Eigen::Vector2i &p_A,
    const MatrixXd &img_A, // expects grayscale
    const MatrixXd &img_B,
    const Eigen::Matrix4d &pose_B_A, // transformation from A to B
    const Matrix3d &K)
{
    assert(img_A.size() == img_B.size());
    double p_z_A = 20;
    cout << K.inverse() << endl;
    Vector3d p_W = p_z_A * (K.inverse() * Vector3d(p_A.x(), p_A.y(), 1)); // eqn. 3.11 [1]
    Eigen::Vector2i p_B = project_pinhole(K * dehomonogenize(pose_B_A * homonogenize(p_W)));
    cout << p_B << endl;
    // plot respective points in image B to validate that correspondence lies on epipolar line
    // double loss = Eigen::abs(img_A(p_A(0), p_A(1)) - img_B(p_B(0), p_B(1)));
}

int main()
{
    cout.precision(2);
    int A_idx = 11 / 2;
    int B_idx = 1;
    MatrixXd img_A, img_B;
    read_image(get_image_file_path(A_idx), img_A);
    read_image(get_image_file_path(B_idx), img_B);

    Eigen::Matrix3d K;
    Eigen::Matrix4d T_a, T_b;
    read_camera_info(get_camera_info_file_path(A_idx), K, T_a);
    read_camera_info(get_camera_info_file_path(B_idx), K, T_b);

    Eigen::Vector2i p(img_A.rows() / 2, img_A.cols() / 2);
    photometric_loss(p, img_A, img_B, T_b * T_a.inverse(), K);
    return 0;
}

/** Bibliography
 * [1] Dense Visual SLAM - R.A. Newcombe, Phd. Thesis
 * https://www.doc.ic.ac.uk/~ajd/Publications/newcombe_phd2012.pdf
 * 
 * [2] DTAM: Dense Tracking and Mapping in Real-Time
 * http://ugweb.cs.ualberta.ca/~vis/courses/CompVis/readings/3DReconstruction/dtam.pdf
*/