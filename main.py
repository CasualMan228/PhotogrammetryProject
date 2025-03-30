#first~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#УКАЗЫВАЙТЕ СВОИ ПУТИ И ПАРАМЕТРЫ!!!

import os
import cv2
import numpy as np
import open3d as o3d

w = 4608
h = 3456
f = 3000 #1,2 * min(h,w)

cx, cy = w/2, h/2

camera = np.array([
    [f, 0, cx],
    [0, f, cy],
    [0, 0, 1]
], dtype=np.float64)
def loadImages(path):
    images = []
    for file in os.listdir(path): #список всех файлов в папке
        if file.endswith(".jpg"):
            pathCurrentImg = os.path.join(path, file) #построение пути
            img = cv2.imread(pathCurrentImg)
            if img is not None:
                images.append(img)
    return images

def detectPoints(images): #ORB -> Oriented Rotated Brief (поиск, обработка корректности обнаружения, фиксация ТОЧЕК)
    orb = cv2.ORB_create()
    points = [] #список точек
    pointsInfo = [] #как выглядят эти точки
    for img in images:
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pointsCurrent, pointsInfoCurrent = orb.detectAndCompute(grayImg, None)
        points.append(pointsCurrent)
        pointsInfo.append(pointsInfoCurrent)
    return points, pointsInfo

def matchPointsInfo(pointsInfo):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #поисковик соответствий пар точек между изображениями и их узор через crossCheck
    matches = []
    for i in range(len(pointsInfo) - 1):
        matchesCurrent = matcher.match(pointsInfo[i],pointsInfo[i+1])
        matchesCurrent = sorted(matchesCurrent, key=lambda x: x.distance)
        matches.append(matchesCurrent)
    return matches

def make3DPoints(points, matches):
    points3D = []
    for i in range(len(matches) - 1):
        points1 = np.array([points[i][m.queryIdx].pt for m in matches[i]]) #сбор в массив точек из первого изображения, где есть совпадения с другим изображением
        points2 = np.array([points[i+1][m.trainIdx].pt for m in matches[i]])
        matrix1 = np.eye(3,4) #матрица камеры, которая по матем. "трюкам"(преобразование матриц) определит положение камеры
        matrix2 = np.hstack((np.eye(3), np.array([[0], [0], [-1]]))) #hstack -> объединить матрицу горизонтально
        points3DMas = cv2.triangulatePoints(matrix1, matrix2, points1.T, points2.T) #T-> превратить строки в столбцы в массивах
        points3DMas /= points3DMas[3] #избавляемся от 4-ой координаты W(нужна была для работы с матрицами)
        points3D.extend(points3DMas.T) #append воспримет элемент как список, а extend - каждый элемент отдельно
    return np.array(points3D)

def createMesh(pcd): #PCD -> облако точек
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=30) #ориентировать уже сформированные нормали к постоянной касательной плоскости (чтобы нормали смотрели в одну сторону)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9) #densities -> плотности точек, depth -> уровень детализации сетки
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox) #также и строчку верхнюю смотрим -> удалить лишние треугольники за пределами бокса, чтобы убрать мусор и артефакты
    return mesh

#~~~

images = loadImages("Images")
points, pointsInfo = detectPoints(images)
matches = matchPointsInfo(pointsInfo)
points3D = make3DPoints(points, matches)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points3D[:, :3]) #выбрать x y z, игнорируя w
o3d.io.write_point_cloud("pointCloud.ply", pcd)

mesh = createMesh(pcd)
o3d.io.write_triangle_mesh("model.obj", mesh)


#second~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import subprocess
def runColmap(commandList):
    subprocess.run(commandList, check=True) #check-> ошибка

runColmap([
    "D:/PythonProjects/PhotogrammetryProject/photogrammetryProject/colmap/COLMAP.bat",
    "feature_extractor", #извлечение ключевых признаков из изображений
    "--database_path", "database.db",
    "--image_path", "Images",
])
runColmap([
    "D:/PythonProjects/PhotogrammetryProject/photogrammetryProject/colmap/COLMAP.bat",
    "exhaustive_matcher", #сопоставление признаков
    "--database_path", "database.db",
])

runColmap([
    "D:/PythonProjects/PhotogrammetryProject/photogrammetryProject/colmap/COLMAP.bat",
    "mapper", #построение sparse-модели (негустой)
    "--image_path", "Images",
    "--database_path", "database.db",
    "--output_path", "sparse"
])

runColmap([
    "D:/PythonProjects/PhotogrammetryProject/photogrammetryProject/colmap/COLMAP.bat",
    "image_undistorter", #устранение искажений изображений
    "--image_path", "Images",
    "--input_path", "sparse/0",
    "--output_path", "dense"
])

runColmap([
    "D:/PythonProjects/PhotogrammetryProject/photogrammetryProject/colmap/COLMAP.bat",
    "patch_match_stereo",  #поиск похожих участков(пикселей) между изображениями
    "--workspace_path", "dense",
    "--workspace_format", "COLMAP"
])

runColmap([
    "D:/PythonProjects/PhotogrammetryProject/photogrammetryProject/colmap/COLMAP.bat",
    "stereo_fusion",  #построение dense-модели (плотный)
    "--workspace_path", "dense",
    "--input_type", "geometric",
    "--output_path", "dense/fused.ply"
])

# runColmap([
#     "D:/PythonProjects/PhotogrammetryProject/photogrammetryProject/colmap/COLMAP.bat",
#     "model_converter", #конвертация в ply
#     "--input_path", "dense/sparse",
#     "--output_path", "model/result.ply",
#     "--output_type", "PLY"
# ])


#third->3DF_Zephyr:/~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~