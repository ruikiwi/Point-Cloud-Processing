# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d
import os
import numpy as np
from pyntcloud import PyntCloud

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX6的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量

def PCA(data, correlation=False, sort=True):
    data = np.array(data)
    data_mean = np.mean(data, axis=0)
    data_centered = data - data_mean
    cov_matrix = np.cov(data_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def main():
    # 加载原始点云
    with open('/Users/cat_ray/Desktop/PointCloud/Chapter1/Homework1/code/modelnet40_normal_resampled/'
              'modelnet40_shape_names.txt') as f:
        a = f.readlines()
    # print('Check 1')
    for i in a:
        if (i != "airplane\n"):
            continue

        point_cloud_pynt = PyntCloud.from_file(
            '/Users/cat_ray/Desktop/PointCloud/Chapter1/Homework1/code/modelnet40_normal_resampled/'
            '{}/{}_0627.txt'.format(i.strip(), i.strip()), sep=",",
            names=["x", "y", "z", "nx", "ny", "nz"])

    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 从点云中获取点，只对点进行处理
    points = np.array(point_cloud_pynt.points)
    points = points[:, :3]     #n×3
    print('total points number is:', points.shape[0], points.shape)

    w, v = PCA(points)
    point_cloud_vector = v[:, 0]
    print('the main orientation of this pointcloud is: ', point_cloud_vector)

    # Project the points onto the 2d plane, and visualize it
    projected_points = np.dot(points, v[:, :2])
    projected_points = np.hstack([projected_points, np.zeros((projected_points.shape[0], 1))])

    projected_point_cloud_o3d = o3d.geometry.PointCloud()
    projected_point_cloud_o3d.points = o3d.utility.Vector3dVector(projected_points)
    o3d.visualization.draw_geometries([projected_point_cloud_o3d])

    # 循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    normals = []

    for point in point_cloud_o3d.points:
        cnt, idxs, dists = pcd_tree.search_knn_vector_3d(point, 10)
        w, v = PCA(points[idxs])
        normals.append(v[:, -1])
    normals = np.array(normals, dtype=np.float64)
    # print("normals", normals.shape)

    normals = np.array(normals, dtype=np.float64)
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)

    o3d.visualization.draw_geometries([point_cloud_o3d], "Open3D normal estimation", width=800, height=600, left=50, top=50,
                                      point_show_normal=True, mesh_show_wireframe=False, mesh_show_back_face=False)


if __name__ == '__main__':
    main()

