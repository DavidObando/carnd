#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose, Point
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial.distance import cdist
import datetime

STATE_COUNT_THRESHOLD = 3
LIGHT_LOCATION_THRESHOLD = 30 # meters, for comparing real position to known position of traffic lights

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # List of positions that correspond to the line to stop in front of for a given intersection
        self.stop_line_positions = self.config['stop_line_positions']

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        self.waypoints_positions = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in self.waypoints.waypoints]

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED or state == TrafficLight.YELLOW else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, position, waypoints=None):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            position (Point): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in waypoints

        """
        # @done implement
        if (not waypoints):
            waypoints = self.waypoints_positions
        return cdist([[position.x, position.y]], waypoints).argmin()

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        light_state = self.light_classifier.get_classification(cv_image, light.state)
        #print "Light state: " + str(light_state) + " at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return light_state
        #return light.state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closest to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # @done find the closest visible traffic light (if one exists)
        if(self.pose and self.waypoints):
            car_position = self.get_closest_waypoint(self.pose.pose.position)
            closest_stop_line_index = self.get_closest_waypoint(self.pose.pose.position, self.stop_line_positions)
            closest_stop_line = self.stop_line_positions[closest_stop_line_index]
            light_wp = self.get_closest_waypoint(Point(closest_stop_line[0], closest_stop_line[1], 0))
            if car_position < (light_wp + 5) and light_wp - car_position < 70: # assume visibility is 70 meters
                for real_light in self.lights:
                    light_position = real_light.pose.pose.position
                    light_x_approx = abs(light_position.x - closest_stop_line[0]) < LIGHT_LOCATION_THRESHOLD
                    light_y_approx = abs(light_position.y - closest_stop_line[1]) < LIGHT_LOCATION_THRESHOLD
                    if light_x_approx and light_y_approx:
                        light = real_light

        if light:
            state = self.get_light_state(light)
            # we're going to decrease the stop index by 5 units given that
            # our classifier works best around the track when it stops at a
            # slightly longer distance from the traffic lights.
            return light_wp - 5, state
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
