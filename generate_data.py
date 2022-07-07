"""
An example of client-side bounding boxes with basic car controls.
Controls:
Welcome to CARLA for Getting Bounding Box Data.
Use WASD keys for control.
    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake
    P            : autopilot mode
    C            : Capture Data
    l            : Loop Capture Start
    L            : Loop Capture End
    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
 #
import math #

# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass

try:
    sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.11-py{}.{}-{}.egg'.format(
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

from carla import ColorConverter as cc

import weakref
import random
import cv2
import time
import argparse
import textwrap

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_TAB
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_p
    from pygame.locals import K_c
    from pygame.locals import K_l
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# draw filtered bounding box
from filter_bb import draw_bb

# spawning vehicles and pedestrians
import spawn_actors

# setting the weather
import set_weather

VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 90
# VIEW_WIDTH = 1440
# VIEW_HEIGHT = 1080
# VIEW_FOV = 50

BB_COLOR = (248, 64, 24)
WBB_COLOR = (0, 0, 255)
vehicle_bbox_record = False
pedestrian_bbox_record = False
count = 0

# rgb_info = np.zeros((540, 960, 3), dtype="i")
# seg_info = np.zeros((540, 960, 3), dtype="i")

rgb_info = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype="i")
seg_info = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype="i")


# Creates Directory
dir_rgb = 'custom_data/'
dir_seg = 'SegmentationImage/'
dir_pbbox = 'PedestrianBBox/'
dir_vbbox = 'VehicleBBox/'
if not os.path.exists(dir_rgb):
    os.makedirs(dir_rgb)
if not os.path.exists(dir_seg):
    os.makedirs(dir_seg)
if not os.path.exists(dir_pbbox):
    os.makedirs(dir_pbbox)
if not os.path.exists(dir_vbbox):
    os.makedirs(dir_vbbox)

# ==============================================================================
# -- PedestrianBoundingBoxes ---------------------------------------------------
# ==============================================================================

class PedestrianBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_bounding_boxes(pedestrians, camera):
        """
        Creates 3D bounding boxes based on carla Pedestrian list and camera.
        """

        bounding_boxes = [PedestrianBoundingBoxes.get_bounding_box(pedestrian, camera) for pedestrian in pedestrians]
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def draw_bounding_boxes(args, display, bounding_boxes):
        """
        Draws bounding boxes on pygame display.
        """
        global pedestrian_bbox_record
        global count

        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))

        if pedestrian_bbox_record == True:
            f = open(f"PedestrianBBox/{args.map}/bbox"+str(count), 'w')
            print("PedestrianBoundingBox")
        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]

            if pedestrian_bbox_record == True:
                f.write(str(points)+"\n")
        
        if pedestrian_bbox_record == True:
            f.close()
            pedestrian_bbox_record = False

        display.blit(bb_surface, (0, 0))

    @staticmethod
    def get_bounding_box(pedestrian, camera):
        """
        Returns 3D bounding box for a pedestrian based on camera view.
        """

        bb_cords = PedestrianBoundingBoxes._create_bb_points(pedestrian)
        cords_x_y_z = PedestrianBoundingBoxes._pedestrian_to_sensor(bb_cords, pedestrian, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _create_bb_points(pedestrian):
        """
        Returns 3D bounding box for a pedestrian.
        """

        cords = np.zeros((8, 4))
        extent = pedestrian.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _pedestrian_to_sensor(cords, pedestrian, sensor):
        """
        Transforms coordinates of a pedestrian bounding box to sensor.
        """

        world_cord = PedestrianBoundingBoxes._pedestrian_to_world(cords, pedestrian)
        sensor_cord = PedestrianBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _pedestrian_to_world(cords, pedestrian):
        """
        Transforms coordinates of a pedestrian bounding box to world.
        """

        bb_transform = carla.Transform(pedestrian.bounding_box.location)
        bb_pedestrian_matrix = PedestrianBoundingBoxes.get_matrix(bb_transform)
        pedestrian_world_matrix = PedestrianBoundingBoxes.get_matrix(pedestrian.get_transform())
        bb_world_matrix = np.dot(pedestrian_world_matrix, bb_pedestrian_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = PedestrianBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix




# ==============================================================================
# -- VehicleBoundingBoxes ---------------------------------------------------
# ==============================================================================


class VehicleBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_bounding_boxes(vehicles, camera):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """
        bounding_boxes = [(VehicleBoundingBoxes.get_bounding_box(vehicle, camera)[0],VehicleBoundingBoxes.get_bounding_box(vehicle, camera)[1])\
             for vehicle in vehicles]
        

        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[0][:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def draw_bounding_boxes(args, display, bounding_boxes):
        """
        Draws bounding boxes on pygame display.
        """
        global vehicle_bbox_record
        global count

        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))


        if vehicle_bbox_record == True:
            f = open(f"VehicleBBox/{args.map}/bbox"+str(count), 'w')
            print("VehicleBoundingBox")
        for idx, bbox in enumerate(bounding_boxes):
            points = [(int(bbox[0][i, 0]), int(bbox[0][i, 1])) for i in range(8)]
            # wheel_nums = int(vehicles[idx].attributes['number_of_wheels'])
            # if wheel_nums == 4:
            #     label = '1'
            # else: label = '0'
            vehicle_type = bbox[1]

            if 'carlacola' in vehicle_type or 'cybertruck' in vehicle_type: # truck
                vehicle_type_id = 1
            
            elif 'gazelle.omafiets' in vehicle_type or 'bh.crossbike' in vehicle_type or 'diamondback.century' in vehicle_type: # bicycle
                vehicle_type_id = 2

            elif 'yamaha.yzf' in vehicle_type or 'kawasaki.ninja' in vehicle_type or 'harley-davidson.low_rider' in vehicle_type: # motorcycle
                vehicle_type_id = 3

            else: # car
                vehicle_type_id = 0

            if vehicle_bbox_record == True:
                f.write(str(points)+'\t'+str(vehicle_type_id)+"\n")
                # f.write(str(points)+"\n")
        
        if vehicle_bbox_record == True:
            f.close()
            vehicle_bbox_record = False
        
        
        
        display.blit(bb_surface, (0, 0))

    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """
        
        bb_cords, vehicle_type = VehicleBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = VehicleBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox, vehicle_type

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """
        # print(vehicle.attributes['number_of_wheels'])
        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        vehicle_type = vehicle.type_id
        # print(vehicle_type)
        # exit(True)
        return cords, vehicle_type 

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = VehicleBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = VehicleBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to worldd.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = VehicleBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = VehicleBoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = VehicleBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix


# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================


class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.camera_segmentation = None
        self.car = None

        self.display = None
        self.image = None
        self.segmentation_image = None

        self.capture = True
        self.capture_segmentation = True

        self.record = True
        self.seg_record = False
        self.rgb_record = False

        self.screen_capture = 0 
        self.loop_state = False 

        # spawn.main()
        # weather.main()

    def camera_blueprint(self, filter):
        """
        Returns camera blueprint.
        """
        
        camera_bp = self.world.get_blueprint_library().find(filter)
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        # Camera distortion
        camera_bp.set_attribute('lens_circle_falloff', str(5.0))
        camera_bp.set_attribute('lens_circle_multiplier', str(0.0)) # 왜곡이 심한 것 확인
        camera_bp.set_attribute('lens_k', str(-1.0))
        camera_bp.set_attribute('lens_kcube', str(0.0))
        camera_bp.set_attribute('lens_x_size', str(0.08))
        camera_bp.set_attribute('lens_y_size', str(0.8))

        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self, spawn_point):
        """
        Spawns actor-vehicle to be controled.
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        # location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, spawn_point)

    def setup_camera(self,args):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """


        seg_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-15))
        self.camera_segmentation = self.world.spawn_actor(self.camera_blueprint('sensor.camera.semantic_segmentation'), seg_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera_segmentation.listen(lambda image_seg: weak_self().set_segmentation(args,weak_self, image_seg))

        #camera_transform = carla.Transform(carla.Location(x=1.5, z=2.8), carla.Rotation(pitch=-15))
        camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-15))
        self.camera = self.world.spawn_actor(self.camera_blueprint('sensor.camera.rgb'), camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(args,weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration
        self.camera_segmentation.calibration = calibration

    def control(self, car):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """
        global keys

        keys = pygame.key.get_pressed()

        if keys[K_ESCAPE]:
            return True

        control = car.get_control()
        control.throttle = 0
        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        if keys[K_p]:
            car.set_autopilot(True)       
        if keys[K_c]:
            self.screen_capture = self.screen_capture + 1
        else:
            self.screen_capture = 0
        if keys[K_l]:
            self.loop_state = True
        if keys[K_l] and (pygame.key.get_mods() & pygame.KMOD_SHIFT):
            self.loop_state = False
        control.hand_brake = keys[K_SPACE]

        car.apply_control(control)
        return False

    @staticmethod
    def set_image(args, weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """
        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

        if self.rgb_record:
            i = np.array(img.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            i3 = i2[:, :, :3]

            cv2.imwrite(f'custom_data/{args.map}/image' + str(self.image_count) + '.png', i3)           
            print("RGB(custom)Image")

    @staticmethod
    def set_segmentation(args, weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture_segmentation:
            self.segmentation_image = img
            self.capture_segmentation = False


        if self.seg_record:
            img.convert(cc.CityScapesPalette)
            i = np.array(img.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            i3 = i2[:, :, :3]

            cv2.imwrite(f'SegmentationImage/{args.map}/seg' + str(self.image_count) +'.png', i3)
            print("SegmentationImage")

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    def game_loop(self, args):
        """
        Main program loop.
        """
        # maps = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD']
        maps = ['Map25']
        for m in maps:
            args.map = m
            

            if not os.path.exists(f"VehicleBBox/{args.map}"):
                os.makedirs(f"VehicleBBox/{args.map}")

            if not os.path.exists(f"custom_data/{args.map}"):
                os.makedirs(f"custom_data/{args.map}")

            if not os.path.exists(f"SegmentationImage/{args.map}"):
                os.makedirs(f"SegmentationImage/{args.map}")

            if not os.path.exists(f"PedestrianBBox/{args.map}"):
                os.makedirs(f"PedestrianBBox/{args.map}")

            try:
                pygame.init()

                self.client = carla.Client('127.0.0.1', 2000)
                self.client.set_timeout(200.0)
                # self.world = self.client.load_world('/Game/Package19/Maps/Map19/Map19')
                # self.world = self.client.get_world()
                self.world = self.client.load_world(args.map)
                
                # set traffic manager
                traffic_manager = self.client.get_trafficmanager(8000) #  디폴트 tm_port: 8000
                traffic_manager.set_global_distance_to_leading_vehicle(1.0)
                traffic_manager.set_hybrid_physics_mode(True)
                traffic_manager.set_random_device_seed(int(time.time()))

                # settings
                settings = self.world.get_settings()
                traffic_manager.set_synchronous_mode(True)
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                self.world.apply_settings(settings)

                # spawning vehicles and pedestrians
                actors = spawn_actors.spawn_actors(self.world, self.client, args, traffic_manager, synchronous_master)
                self.world, self.client, vehicles_list, walkers_list, all_actors, actor_spawn_point, all_id = actors.spawn()

                # set weather
                speed_factor = args.speed
                update_freq = 0.1 / speed_factor
                weather = set_weather.Weather(self.world.get_weather(), args)
                elapsed_time = 0.0

                # setup my car & camera
                self.setup_car(actor_spawn_point)
                self.setup_camera(args)

                self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
                pygame_clock = pygame.time.Clock()

                self.set_synchronous_mode(True)

                vehicles = self.world.get_actors().filter('vehicle.*')
                pedestrians = self.world.get_actors().filter('walker.pedestrian.*')

                if args.autopilot:
                    self.car.set_autopilot(True)

                self.image_count = 0
                self.time_interval = 0

                global vehicle_bbox_record
                global pedestrian_bbox_record
                global count
                data_count = 0
                while data_count < args.data:
                    # self.world.tick()
                    self.capture = True
                    pygame_clock.tick_busy_loop(60) 

                    self.render(self.display)
                    
                    self.time_interval += 1
                    self.loop_state = True
                    if ((self.time_interval % args.CaptureLoop) == 0 and self.loop_state):
                        self.image_count = self.image_count + 1 
                        self.rgb_record = True
                        self.seg_record = True
                        vehicle_bbox_record = True
                        pedestrian_bbox_record = True
                        count = self.image_count
                        weather.tick(3)
                        self.world.set_weather(weather.weather)
                        sys.stdout.write('\r' + str(weather) + 12 * ' ')
                        print("-------------------------------------------------")
                        print("ImageCount - %d" %self.image_count)
                        data_count += 1

                    if self.screen_capture == 1:
                        self.image_count = self.image_count + 1 
                        self.rgb_record = True
                        self.seg_record = True
                        vehicle_bbox_record = True
                        pedestrian_bbox_record = True
                        count = self.image_count
                        weather.tick(3)
                        self.world.set_weather(weather.weather)
                        sys.stdout.write('\r' + str(weather) + 12 * ' ')
                        print("-------------------------------------------------")
                        print("Captured! ImageCount - %d" %self.image_count)
                        data_count += 1
                        
                    self.world.tick()

                    bounding_boxes = VehicleBoundingBoxes.get_bounding_boxes(vehicles, self.camera)
                    pedestrian_bounding_boxes = PedestrianBoundingBoxes.get_bounding_boxes(pedestrians, self.camera)

                    VehicleBoundingBoxes.draw_bounding_boxes(args, self.display, bounding_boxes)
                    PedestrianBoundingBoxes.draw_bounding_boxes(args, self.display, pedestrian_bounding_boxes)
                    
                    time.sleep(0.03)

                    self.rgb_record = False
                    self.seg_record = False
                    pygame.display.flip()

                    pygame.event.pump()
                    
                    if self.control(self.car):
                        return

                    data_list = os.listdir(f'./custom_data/{args.map}')
                    # data_count = len(data_list)
                    
                    # while True:
                    #     self.world.wait_for_tick()

            finally:
                print('Draw filtered bounding box')
                bb = draw_bb(args)
                bb.make_filtered_bb()

                if synchronous_master:
                    settings = self.world.get_settings()
                    settings.synchronous_mode = False
                    settings.fixed_delta_seconds = None
                    self.world.apply_settings(settings)
                print('\ndestroying %d vehicles' % len(vehicles_list))
                self.client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

                # stop walker controllers (list is [controller, actor, controller, actor ...])
                for i in range(0, len(all_id), 2):
                    all_actors[i].stop()

                print('\ndestroying %d walkers' % len(walkers_list))
                self.client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

                time.sleep(0.5)
                self.set_synchronous_mode(False)
                self.camera.destroy()
                self.camera_segmentation.destroy()
                self.car.destroy()
                pygame.quit()




# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    """
    Initializes the client-side bounding box demo.
    """
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-l', '--CaptureLoop',
        metavar='N',
        default=20,
        type=int,
        help='set Capture Cycle settings, Recommand : above 100')

    argparser.add_argument(
        '-m', '--map',
        default='Map25',
        type=str,
        help='map name')

    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=10,
        type=int,
        help='number of vehicles (default: 60)')

    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=10,
        type=int,
        help='number of walkers (default: 60)')

    argparser.add_argument(
        '-s', '--speed',
        metavar='FACTOR',
        default=1.0,
        type=float,
        help='rate at which the weather changes (default: 1.0)')

    argparser.add_argument(
        '-d', '--data',
        default=10,
        type=int,
        help='number of data per one map')

    argparser.add_argument(
        '-p', '--autopilot',
        action='store_true',
        help='set autopilot')

    argparser.add_argument(
        '-st', '--storm',
        action='store_true',
        help='enable storm weather')

    args = argparser.parse_args()
    
    print(__doc__)

    
    
    try:
        client = BasicSynchronousClient()
        client.game_loop(args)
    finally:

        print('EXIT')
    


if __name__ == '__main__':
    main()
