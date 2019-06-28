#include "kinematics.h"

kinematics::kinematics()
{

}

arr kinematics::ik_compute(rai::KinematicWorld &kine_world, RobotOperation robot_op,
               arr &target_position, arr q_home, bool sending_motion, bool verbose) {
    rai::Frame *objectFrame = kine_world.addFrame("obj");
    objectFrame->setShape(rai::ST_ssBox, {.1, .1, .1, .02});
    objectFrame->setColor({.8, .8, .1});
    objectFrame->setPosition(target_position);

    double tolerate=0.0001;
    double time=4.0;

    // tracking IK
    arr y, y_verbose, J, Phi, PhiJ;
    arr q, q_best;
    int best_iter;
    double best_error = 100.0;
    arr Wmetric = diag(2., kine_world.getJointStateDimension());

    cout<<"IK: target postion at "<<target_position<<endl;


    int i = 0;
    while(i < 1000) {
        Phi.clear();
        PhiJ.clear();

        //1st task: go to target pos
        kine_world.evalFeature(y, J, FS_position, {"pointer"});
        arr pos_diff = y-target_position;
        //pos_diff(2) *= 1e1; // emphasize on z coord
        Phi.append( (pos_diff) * 1e2);
        PhiJ.append( J * 1e2 );

        //2nd task: joint should stay close to zero
        kine_world.evalFeature(y, J, FS_qItself, {});
        Phi .append( (y-q_home) * 1e0 );
        PhiJ.append( J * 1e0 );

        //3rd task: joint angles
        kine_world.evalFeature(y, J, FS_vectorZDiff, {"pointer", "obj"});
        Phi.append( y * 1e1);
        PhiJ.append( J * 1e1 );

        // IK compute joint updates
        q = kine_world.getJointState();
        q -= 0.05*inverse(~PhiJ*PhiJ + Wmetric) * ~PhiJ * Phi;

        kine_world.setJointState(q);
        kine_world.watch();

        // verbose
        if (verbose) {
            kine_world.evalFeature(y, J, FS_position, {"pointer"});
            cout << "iter " << i+1 << " pos diff = " << y-target_position << endl;
            kine_world.evalFeature(y_verbose, J, FS_position, {"pointer"});
            cout << "     current position="<<y_verbose<<", target position="<<target_position<<endl;
            kine_world.evalFeature(y_verbose, J, FS_quaternion, {"baxterR"});
            cout << "     current quaternion="<<y_verbose<<", target quaternion="<<objectFrame->getQuaternion()<<endl;
            cout << "     abs error=" << sumOfAbs(y-target_position)/3.0<<endl;
            cout << "     phi and phi j sizes="<<Phi.N<<" "<<PhiJ.N<<endl;
        } else kine_world.evalFeature(y, J, FS_position, {"pointer"});

        // save best motion
        double error = sumOfAbs(y-target_position)/3.0;
        if (error < best_error) {
            best_error = error;
            q_best = q;
            best_iter = i;
        }

        // evaluate to terminate early
        if (error < tolerate) break;
        i++;
    }

    printf("IK: done in %d iters with error=%f at iter %d\n", i, best_error, best_iter);
    q_best(-2) = 0;
    if (sending_motion) robot_op.move({q_best}, {time});
    kine_world.setJointState(q_best);

    kine_world.evalFeature(y, J, FS_position, {"pointer"});
    cout<<"IK: final postion at "<<y<<endl;

    kine_world.evalFeature(y_verbose, J, FS_vectorZ, {"pointer"});
    cout<<"IK: final z vector = "<<y_verbose;
    kine_world.evalFeature(y_verbose, J, FS_quaternion, {"pointer"});
    cout<<" final quat = "<<y_verbose<<endl;

    kine_world.evalFeature(y_verbose, J, FS_vectorZ, {"obj"});
    cout<<"IK: target z vector = "<<y_verbose;
    kine_world.evalFeature(y_verbose, J, FS_quaternion, {"obj"});
    cout<<" target quat = "<<y_verbose<<endl;

    kine_world.evalFeature(y_verbose, J, FS_vectorZDiff, {"pointer", "obj"});
    cout<<"IK: Z vector diff = "<<y_verbose<<" abs error = "<<sumOfAbs(y_verbose)<<endl;

    kine_world.evalFeature(y_verbose, J, FS_quaternionDiff, {"pointer", "obj"});
    cout<<"IK: quaternion diff = "<<y_verbose<<" abs error = "<<sumOfAbs(y_verbose)<<endl;

    return q_best;
}

arr kinematics::ik_compute_with_grabbing(rai::KinematicWorld &kine_world, RobotOperation robot_op,
                             arr &target_position, arr q_home, bool sending_motion) {
    rai::Frame *objectFrame = kine_world.addFrame("obj");
    objectFrame->setShape(rai::ST_ssBox, {.1, .1, .1, .02});
    objectFrame->setColor({.8, .8, .1});
    objectFrame->setPosition(target_position);

    double tolerate=0.01;
    double time=4.0;

    // tracking IK
    arr y, y_verbose, J, Phi, PhiJ;
    arr q, q_best;
    int best_iter;
    double best_error = 100.0;
    arr Wmetric = diag(2., kine_world.getJointStateDimension());

    cout<<"IK: target postion at "<<target_position<<endl;


    int i = 0;
    while(i < 200) {
        Phi.clear();
        PhiJ.clear();

        //1st task: go to target pos
        kine_world.evalFeature(y, J, FS_position, {"pointer"});
        arr pos_diff = y-target_position;
        //pos_diff(2) *= 1e1; // emphasize on z coord
        Phi.append( (pos_diff) * 1e2);
        PhiJ.append( J * 1e2 );

        //2nd task: joint should stay close to zero
        kine_world.evalFeature(y, J, FS_qItself, {});
        Phi .append( (y-q_home) * 1e0 );
        PhiJ.append( J * 1e0 );

        //3rd task: joint angles
        kine_world.evalFeature(y, J, FS_vectorZDiff, {"pointer", "obj"});
        Phi.append( y * 1e1);
        PhiJ.append( J * 1e1 );

        // IK compute joint updates
        q = kine_world.getJointState();
        q -= 0.05*inverse(~PhiJ*PhiJ + Wmetric) * ~PhiJ * Phi;

        kine_world.setJointState(q);
        kine_world.watch();

        kine_world.evalFeature(y, J, FS_position, {"pointer"});

        // save best motion
        double error = sumOfAbs(y-target_position)/3.0;
        if (error < best_error) {
            best_error = error;
            q_best = q;
            best_iter = i;
        }

        // evaluate to terminate early
        if (error < tolerate) break;
        i++;
    }

    printf("IK: done in %d iters with error=%f at iter %d\n", i, best_error, best_iter);
    q_best(-2) = 1;
    if (sending_motion) robot_op.move({q_best}, {time});
    kine_world.setJointState(q_best);

    kine_world.evalFeature(y, J, FS_position, {"pointer"});
    cout<<"IK: final postion at "<<y<<endl;

    kine_world.evalFeature(y_verbose, J, FS_vectorZ, {"pointer"});
    cout<<"IK: final z vector = "<<y_verbose;
    kine_world.evalFeature(y_verbose, J, FS_quaternion, {"pointer"});
    cout<<" final quat = "<<y_verbose<<endl;

    kine_world.evalFeature(y_verbose, J, FS_vectorZ, {"obj"});
    cout<<"IK: target z vector = "<<y_verbose;
    kine_world.evalFeature(y_verbose, J, FS_quaternion, {"obj"});
    cout<<" target quat = "<<y_verbose<<endl;

    kine_world.evalFeature(y_verbose, J, FS_vectorZDiff, {"pointer", "obj"});
    cout<<"IK: Z vector diff = "<<y_verbose<<" abs error = "<<sumOfAbs(y_verbose)<<endl;

    kine_world.evalFeature(y_verbose, J, FS_quaternionDiff, {"pointer", "obj"});
    cout<<"IK: quaternion diff = "<<y_verbose<<" abs error = "<<sumOfAbs(y_verbose)<<endl;

    return q_best;
}

rai::KinematicWorld kinematics::setup_kinematic_world() {
    rai::KinematicWorld C;
    C.addFile("../rai-robotModels/baxter/baxter_new.g");

    // add a frame for the camera
    rai::Frame *cameraFrame = C.addFrame("camera", "head");
    cameraFrame->Q.setText("d(-90 0 0 1) t(-.08 .205 .115) d(26 1 0 0) d(-1 0 1 0) d(6 0 0 1)");
    cameraFrame->calc_X_from_parent();
    cameraFrame->setPosition({-0.0472772, 0.226517, 1.79207});
    cameraFrame->setQuaternion({0.969594, 0.24362, -0.00590741, 0.0223832});

    // add a frame for the object
    rai::Frame *objectFrame = C.addFrame("obj");
    objectFrame->setShape(rai::ST_ssBox, {.1, .1, .1, .02});
    objectFrame->setColor({.8, .8, .1});

    // add a frame for the endeff reference
    rai::Frame *pointerFrame = C.addFrame("pointer", "baxterR");
    pointerFrame->setShape(rai::ST_ssBox, {.05, .05, .05, .01});
    pointerFrame->setColor({.8, .1, .1});
    pointerFrame->setRelativePosition({0.,0.,-.05});

    return C;
}

