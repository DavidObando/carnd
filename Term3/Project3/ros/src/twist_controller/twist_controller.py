import rospy

from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter
import math


class YawControllerProperties:
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.min_speed = min_speed
        self.max_lat_accel = max_lat_accel
        self.max_steer_angle = max_steer_angle


class PIDControllerProperties:
    def __init__(self, decel_limit, accel_limit, max_brake_torque, max_throttle):
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.max_brake_torque = max_brake_torque
        self.max_throttle = max_throttle


class LowPassFilterProperties:
    def __init__(self, wheel_radius, brake_deadband):
        self.wheel_radius = wheel_radius
        self.brake_deadband = brake_deadband
        self.ts = 1.0/50.0


class Controller(object):
    def __init__(self, pid_controller_properties, yaw_controller_properties, lowpass_filter_properties):

        # @done Find suitable values for PID properties
        self.pid = PID(kp=2.0, ki=0.05, kd=0.005,
                       mn=pid_controller_properties.decel_limit,
                       mx=pid_controller_properties.accel_limit)

        self.max_brake_torque = pid_controller_properties.max_brake_torque
        self.max_throttle = pid_controller_properties.max_throttle

        self.yaw_controller = YawController(yaw_controller_properties.wheel_base,
                                            yaw_controller_properties.steer_ratio,
                                            yaw_controller_properties.min_speed,
                                            yaw_controller_properties.max_lat_accel,
                                            yaw_controller_properties.max_steer_angle)

        # @done Check what are actual correct properties for tau and ts for the lowpass filter
        self.low_pass_filter_pid = LowPassFilter(lowpass_filter_properties.brake_deadband,
                                                 lowpass_filter_properties.ts)

        self.low_pass_filter_yaw_controller = LowPassFilter(lowpass_filter_properties.wheel_radius,
                                                            lowpass_filter_properties.ts)


    def control(self, proposed_linear_velocity, proposed_angular_velocity, current_linear_velocity, is_dbw_enabled):
        throttle = 0.0
        brake = 0.0
        steer = 0.0

        if not is_dbw_enabled:
            rospy.loginfo("Gauss - DBW not enabled, resetting PID, throttle, brake and steer")
            self.pid.reset()
            return throttle, brake, steer

        # Compute the error for the PID
        error = proposed_linear_velocity - current_linear_velocity
        rospy.loginfo("Gauss - Got error for PID: " + str(error))

        pid_result = self.pid.step(error=error, sample_time=1.0/50.0)
        rospy.loginfo("Gauss - PID Result: " + str(pid_result))

        stopdeadzone = 0.5
        if pid_result >= 0.0 and proposed_linear_velocity >= stopdeadzone:
            # We want to accelerate
            throttle = self.scale(pid_result, self.max_throttle)
            brake = 0.0
        else:
            # We want to decelerate
            throttle = 0.0
            brake = self.scale(abs(pid_result), self.max_brake_torque)

            # Apply low pass filter on braking
            brake = self.low_pass_filter_pid.filt(brake)

        steer = self.yaw_controller.get_steering(proposed_linear_velocity,
                                                 proposed_angular_velocity,
                                                 current_linear_velocity)

        # Apply low pass filter on steering
        kSteeringFactor = 1.2
        steer = self.low_pass_filter_yaw_controller.filt(steer) * kSteeringFactor

        rospy.loginfo("Gauss - Throttle: " + str(throttle))
        rospy.loginfo("Gauss - Brake: " + str(brake))
        rospy.loginfo("Gauss - Steering: " + str(steer))

        # Return throttle, brake, steer
        return throttle, brake, steer


    def scale(self, value, scale):
        kStretch = 0.5
        if value < 0.0:
            return 0.0
        else:
            return scale * math.tanh(value * kStretch)
