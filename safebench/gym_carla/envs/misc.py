''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 17:19:24
Description: 
    Copyright (c) 2022-2023 Safebench Team

    Modified from <https://github.com/cjy1992/gym-carla/blob/master/gym_carla/envs/misc.py>
    Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import math
import numpy as np
import carla
import pygame
from matplotlib.path import Path
import skimage
import cv2


def get_speed(vehicle):
    """
        Compute speed of a vehicle in Kmh
        :param vehicle: the vehicle for which speed is calculated
        :return: speed as a float in Kmh
    """
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def get_pos(vehicle):
    """
        Get the position of a vehicle
        :param vehicle: the vehicle whose position is to get
        :return: speed as a float in Kmh
    """
    trans = vehicle.get_transform()
    x = trans.location.x
    y = trans.location.y
    return x, y


def get_info(vehicle):
    """
        Get the full info of a vehicle
        :param vehicle: the vehicle whose info is to get
        :return: a tuple of x, y positon, yaw angle and half length, width of the vehicle
    """
    trans = vehicle.get_transform()
    x = trans.location.x
    y = trans.location.y
    yaw = trans.rotation.yaw / 180 * np.pi
    bb = vehicle.bounding_box
    l = bb.extent.x
    w = bb.extent.y
    info = (x, y, yaw, l, w)
    return info


def get_local_pose(global_pose, ego_pose):
    """
        Transform vehicle to ego coordinate
        :param global_pose: surrounding vehicle's global pose
        :param ego_pose: ego vehicle pose
        :return: tuple of the pose of the surrounding vehicle in ego coordinate
    """
    x, y, yaw = global_pose
    ego_x, ego_y, ego_yaw = ego_pose
    R = np.array([[np.cos(ego_yaw), np.sin(ego_yaw)],
                                [-np.sin(ego_yaw), np.cos(ego_yaw)]])
    vec_local = R.dot(np.array([x - ego_x, y - ego_y]))
    yaw_local = yaw - ego_yaw
    local_pose = (vec_local[0], vec_local[1], yaw_local)
    return local_pose


def get_pixel_info(local_info, d_behind, obs_range, image_size):
    """
        Transform local vehicle info to pixel info, with ego placed at lower center of image.
        Here the ego local coordinate is left-handed, the pixel coordinate is also left-handed,
        with its origin at the left bottom.
        :param local_info: local vehicle info in ego coordinate
        :param d_behind: distance from ego to bottom of FOV
        :param obs_range: length of edge of FOV
        :param image_size: size of edge of image
        :return: tuple of pixel level info, including (x, y, yaw, l, w) all in pixels
    """
    x, y, yaw, l, w = local_info
    x_pixel = (x + d_behind) / obs_range * image_size
    y_pixel = y / obs_range * image_size + image_size / 2
    yaw_pixel = yaw
    l_pixel = l / obs_range * image_size
    w_pixel = w / obs_range * image_size
    pixel_tuple = (x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel)
    return pixel_tuple


def get_poly_from_info(info):
    """
        Get polygon for info, which is a tuple of (x, y, yaw, l, w) in a certain coordinate
        :param info: tuple of x,y position, yaw angle, and half length and width of vehicle
        :return: a numpy array of size 4x2 of the vehicle rectangle corner points position
    """
    x, y, yaw, l, w = info
    poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
    return poly


def get_pixels_inside_vehicle(pixel_info, pixel_grid):
    """
        Get pixels inside a vehicle, given its pixel level info (x, y, yaw, l, w)
        :param pixel_info: pixel level info of the vehicle
        :param pixel_grid: pixel_grid of the image, a tall numpy array pf x, y pixels
        :return: the pixels that are inside the vehicle
    """
    poly = get_poly_from_info(pixel_info)
    p = Path(poly)    # make a polygon
    grid = p.contains_points(pixel_grid)
    isinPoly = np.where(grid == True)
    pixels = np.take(pixel_grid, isinPoly, axis=0)[0]
    return pixels


def get_lane_dis(waypoints, x, y):
    """
        Calculate distance from (x, y) to waypoints.
        :param waypoints: a list of list storing waypoints like [[x0, y0], [x1, y1], ...]
        :param x: x position of vehicle
        :param y: y position of vehicle
        :return: a tuple of the distance and the closest waypoint orientation
    """
    eps = 1e-5
    dis_min = 99999
    waypt = waypoints[0]
    for pt in waypoints:
        d = np.sqrt((x-pt[0])**2 + (y-pt[1])**2)
        if d < dis_min:
            dis_min = d
            waypt = pt
    vec = np.array([x - waypt[0], y - waypt[1]])
    lv = np.linalg.norm(np.array(vec)) + eps
    w = np.array([np.cos(waypt[2]/180*np.pi), np.sin(waypt[2]/180*np.pi)])
    cross = np.cross(w, vec/lv)
    dis = - lv * cross
    return dis, w


def get_preview_lane_dis(waypoints, x, y, idx=2):
    """
        Calculate distance from (x, y) to a certain waypoint
        :param waypoints: a list of list storing waypoints like [[x0, y0], [x1, y1], ...]
        :param x: x position of vehicle
        :param y: y position of vehicle
        :param idx: index of the waypoint to which the distance is calculated
        :return: a tuple of the distance and the waypoint orientation
    """
    waypt = waypoints[idx]
    vec = np.array([x - waypt[0], y - waypt[1]])
    lv = np.linalg.norm(np.array(vec))
    w = np.array([np.cos(waypt[2]/180*np.pi), np.sin(waypt[2]/180*np.pi)])
    cross = np.cross(w, vec/lv)
    dis = - lv * cross
    return dis, w


def is_within_distance_ahead(target_location, current_location, orientation, max_distance):
    """
        Check if a target object is within a certain distance in front of a reference object.

        :param target_location: location of the target object
        :param current_location: location of the reference object
        :param orientation: orientation of the reference object
        :param max_distance: maximum allowed distance
        :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)
    if norm_target > max_distance:
        return False

    forward_vector = np.array(
        [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

    return d_angle < 90.0


def compute_magnitude_angle(target_location, current_location, orientation):
    """
        Compute relative angle and distance between a target_location and a current_location

        :param target_location: location of the target object
        :param current_location: location of the reference object
        :param orientation: orientation of the reference object
        :return: a tuple composed by the distance to the object and the angle between both objects
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

    return (norm_target, d_angle)


def distance_vehicle(waypoint, vehicle_transform):
    loc = vehicle_transform.location
    dx = waypoint.transform.location.x - loc.x
    dy = waypoint.transform.location.y - loc.y

    return math.sqrt(dx * dx + dy * dy)


def set_carla_transform(pose):
    """
        Get a carla transform object given pose.
        :param pose: list if size 3, indicating the wanted [x, y, yaw] of the transform
        :return: a carla transform object
    """
    transform = carla.Transform()
    transform.location.x = pose[0]
    transform.location.y = pose[1]
    transform.rotation.yaw = pose[2]
    return transform

def display_to_rgb(display, obs_size):
    """
        Transform image grabbed from pygame display to an rgb image uint8 matrix
        :param display: pygame display input
        :param obs_size: rgb image size
        :return: rgb image uint8 matrix
    """
    rgb = np.fliplr(np.rot90(display, 3))    # flip to regular view
    rgb = skimage.transform.resize(rgb, (obs_size, obs_size))    # resize
    rgb = rgb * 255
    return rgb


def rgb_to_display_surface(rgb, display_size):
    """
        Generate pygame surface given an rgb image uint8 matrix
        :param rgb: rgb image uint8 matrix
        :param display_size: display size
        :return: pygame surface
    """
    surface = pygame.Surface((display_size, display_size)).convert()
    display = skimage.transform.resize(rgb, (display_size, display_size))
    display = np.flip(display, axis=1)
    display = np.rot90(display, 1)
    pygame.surfarray.blit_array(surface, display)
    return surface


def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2]


def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def debug_bbox(data, obs_size, fov, camera_sensor, world, ego):
    edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
    K = build_projection_matrix(obs_size, obs_size, fov)
    world_2_camera = np.array(camera_sensor.get_transform().get_inverse_matrix())

    signs = world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)
    for bbox in signs:
        if bbox.location.distance(ego.get_transform().location) < 50:
            forward_vec = ego.get_transform().get_forward_vector()
            ray = bbox.location - ego.get_transform().location
            if forward_vec.dot(ray) > 5.0:
                verts = [v for v in bbox.get_world_vertices(carla.Transform())]
                for edge in edges:
                    p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                    p2 = get_image_point(verts[edge[1]], K, world_2_camera)
                    cv2.line(data, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,0,255, 255), 2)
                    signs = world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)

    signs = world.get_level_bbs(carla.CityObjectLabel.Vehicles)                
    for bbox in signs:
        if bbox.location.distance(ego.get_transform().location) < 50:
            forward_vec = ego.get_transform().get_forward_vector()
            ray = bbox.location - ego.get_transform().location
            if forward_vec.dot(ray) > 5.0:
                verts = [v for v in bbox.get_world_vertices(carla.Transform())]
                for edge in edges:
                    p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                    p2 = get_image_point(verts[edge[1]], K, world_2_camera)
                    cv2.line(data, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,0,255, 255), 2)
    return data