#!/usr/bin/env python

import sys
import math
import rospy

from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial.distance import cdist
from std_msgs.msg import Int32


'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

Stopline location for each traffic light.
'''

kLookAheadWaypoints = 200 # Number of waypoints we will publish. You can change this number
kComfortBraking = 1.5 # m/s**2
kFullStopDistance = 2.0

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater', log_level=rospy.DEBUG)
        rospy.loginfo("Gauss - Started Waypoint Updater")

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)

        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        self.base_waypoints_msg = None
        self.waypt_count = 0

        self.lightidx = -1          # Waypoint of last set traffic light to stop at (-1 for none)
        self.obstacleidx = -1       # Waypoint of last set obstacle detected (-1 for none)

        self.targetvel = self.kmph2mps(rospy.get_param("/waypoint_loader/velocity"))

        self.current_velocity = None
        self.position = None

        rate = rospy.Rate(10) # in Hz
        while not rospy.is_shutdown():
            self.update_final_waypoints()
            rate.sleep()


    def update_final_waypoints(self):
        if not self.position or not self.current_velocity or not self.base_waypoints_msg:
            return

        index = self.closest_waypoint_index(self.position)

        highval = sys.maxint
        lightconv = highval if self.lightidx == -1 else self.lightidx
        obstcconv = highval if self.obstacleidx == -1 else self.obstacleidx

        stopidx = min(lightconv, obstcconv)
        end_wpt_write = index+kLookAheadWaypoints

        all_waypoints = self.base_waypoints_msg.waypoints
        current_waypoint_velocity = self.get_waypoint_velocity(all_waypoints[index])
        rospy.logdebug("Stop - Current Waypoint Velocity: " + str(current_waypoint_velocity))

        if stopidx != highval:
            distance_to_stop = self.distance(all_waypoints, index, stopidx)
            rospy.logdebug("Stop - Distance to stop: " + str(distance_to_stop))

            current_actual_velocity = self.current_velocity
            distance_to_start_braking_wrt_waypoint_velocity = 0.5 * (current_waypoint_velocity**2) / kComfortBraking
            distance_to_start_braking_wrt_actual_velocity = 0.5 * (current_actual_velocity**2) / kComfortBraking
            distance_to_start_braking = max(distance_to_start_braking_wrt_waypoint_velocity,
                                            distance_to_start_braking_wrt_actual_velocity)
            rospy.logdebug("Stop - Distance to start braking: " + str(distance_to_start_braking))

            diff_in_waypoint_indices = max(0, stopidx - index)
            rospy.logdebug("Stop - Diff in waypoint indices: " + str(diff_in_waypoint_indices))

            max_allowed_speed = math.sqrt(2.0 * kComfortBraking * distance_to_stop)
            rospy.logdebug("Stop - Max allowed speed: " + str(max_allowed_speed))

            if (diff_in_waypoint_indices != 0):
                if distance_to_stop <= distance_to_start_braking:
                    diff_in_velocity = current_waypoint_velocity / diff_in_waypoint_indices
                    rospy.logdebug("Stop - Diff in velocity: " + str(diff_in_velocity))

                    for i, wpt in enumerate(range(index, stopidx)):
                        if distance_to_stop > kFullStopDistance:
                            target_velocity = min(max_allowed_speed, current_waypoint_velocity - ((i+1) * diff_in_velocity))
                        else:
                            target_velocity = 0.0
                        rospy.logdebug("Stop - Target velocity: " + str(target_velocity))
                        self.set_waypoint_velocity(all_waypoints, wpt % self.waypt_count, target_velocity)
            else:
                # We passed the traffic light, perform a full brake
                target_velocity = 0.0
                for wpt in range(index, stopidx):
                    self.set_waypoint_velocity(all_waypoints, wpt % self.waypt_count, target_velocity)
                    rospy.logdebug("Stop - Performing full brake: " + str(target_velocity))

        else:
            for wpt in range(index, end_wpt_write):
                target_velocity = self.targetvel
                self.set_waypoint_velocity(all_waypoints, wpt % self.waypt_count, target_velocity)

        waypoints_sliced = self.base_waypoints_msg.waypoints[index:index+kLookAheadWaypoints]
        if end_wpt_write >= self.waypt_count:
            waypoints_sliced += self.base_waypoints_msg.waypoints[0: end_wpt_write - self.waypt_count]

        output_msg = Lane()
        output_msg.header = self.base_waypoints_msg.header
        output_msg.waypoints = waypoints_sliced

        rospy.logdebug("Gauss - Publishing Waypoints of length: " + str(len(output_msg.waypoints)))
        self.final_waypoints_pub.publish(output_msg)


    def pose_cb(self, msg):
        self.position = msg.pose.position
        x = self.position.x
        y = self.position.y
        rospy.logdebug("Gauss - Got Pose (x, y): " + str(x) + ", " + str(y))


    def current_velocity_cb(self, msg):
        self.current_velocity = msg.twist.linear.x


    def waypoints_cb(self, waypoints):
        global kLookAheadWaypoints

        rospy.logdebug("Gauss - Got Waypoints")
        self.base_waypoints_msg = waypoints
        self.waypoints_positions = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
        self.waypt_count = len(self.waypoints_positions)

        # In the highly unlikely situation that we have a higher
        # kLookAheadWaypoints than waypoints, the wrap-around writing of
        # waypoints would start overwriting itself.
        kLookAheadWaypoints = min(kLookAheadWaypoints, self.waypt_count)


    def closest_waypoint_index(self, position, waypoints=None):
        if (not waypoints):
            waypoints = self.waypoints_positions
        return cdist([[position.x, position.y]], waypoints).argmin()


    def traffic_cb(self, msg):
        if self.lightidx != msg.data:
            self.lightidx = msg.data


    def obstacle_cb(self, msg):
        if self.obstacleidx != msg.data:
            self.obstacleidx = msg.data


    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x


    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity


    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


    def kmph2mps(self, velocity_kmph):
        return (velocity_kmph * 1000.) / (60. * 60.)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
