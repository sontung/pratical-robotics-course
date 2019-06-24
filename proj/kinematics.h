#ifndef KINEMATICS_H
#define KINEMATICS_H

#include <RosCom/baxter.h>
#include <Kin/frame.h>
#include <Gui/opengl.h>
#include <Operate/robotOperation.h>
#include <unistd.h>

class kinematics
{
public:
    kinematics();
    static arr ik_compute(rai::KinematicWorld &kine_world, RobotOperation robot_op,
                   arr &target_position, arr q_home, bool sending_motion=true,
                   bool verbose=false);
    static arr ik_compute_with_grabbing(rai::KinematicWorld &kine_world, RobotOperation robot_op,
                                 arr &target_position, arr q_home, bool sending_motion=true);
    static rai::KinematicWorld setup_kinematic_world();
};

#endif // KINEMATICS_H
