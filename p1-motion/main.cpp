#include <Kin/kin.h>
#include <Kin/frame.h>

#include <RosCom/baxter.h>
#include <Operate/robotOperation.h>

void minimal_use(){
        rai::KinematicWorld K;
    K.addFile("../../rai-robotModels/baxter/baxter.g");
	arr q0 = K.getJointState();

	BaxterInterface B(true);
	B.send_q(q0);

	for(uint i=0;i<10;i++){
		rai::wait(.1);
		cout <<B.get_q() <<endl;
		cout <<B.get_qdot() <<endl;
		cout <<B.get_u() <<endl;
	}
	K.watch(true);

	arr q = q0;
	q = 0.;
	K.setJointState(q);
	B.send_q(q);
	K.watch(true);
}


arr ik_algo(rai::KinematicWorld kine_world, arr q_home, RobotOperation robot) {
	arr pose_diff;
	arr Phi, PhiJ, J;
	arr q_start = 1.0*q_home;
	
	arr W;
	uint n = kine_world.getJointStateDimension();
	double w = rai::getParameter("w",1e-4);
	W.setDiag(w,n);  //W is equal the Id_n matrix times scalar w
//W.N //total
//W.d0 //#rows
//W.d1 //# cols
	
	StringA null_motion_joints = {"left_s0", "left_s1", "left_e0", "left_e1", "left_w0", "left_w1"};
    std::vector<arr> null_poses;
	for (int j=0; j<6; j++) {
		arr pose, Jtrash;
		kine_world.evalFeature(pose, Jtrash, FS_pose, {null_motion_joints(j)});
		null_poses.push_back(pose);
	}
	robot.move({q_start}, {5});

	
	for (int i=0; i<10; i++) {
		kine_world.getJointState(q_start);
		kine_world.evalFeature(pose_diff, J, FS_poseDiff, {"left_w2", "object"});
		//K.evalFeature(posDiff, Jtrash, FS_positionDiff, {"right_w2", "object"});
		//K.evalFeature(qDiff, Jtrash, FS_quaternionDiff, {"right_w2", "object"});
		//K.evalFeature(y, J, FS_position, {"right_w2"});
		Phi.clear();
		PhiJ.clear();

		pose_diff(0) += 0.3;

		Phi.append(pose_diff);
		PhiJ.append( J );
		
		for(int n=0; n<6; n++)  {
			arr y_null, J_null;
			kine_world.evalFeature(y_null, J_null, FS_pose, {null_motion_joints(n)});
			Phi.append((y_null-null_poses[n])*0.1);
		    PhiJ.append( J_null*0.1 );
		}
		
		//cout <<"q home: " <<q_home<<endl;
		q_start -= inverse(~PhiJ*PhiJ + W)*~PhiJ*Phi;
		//q_home += inverse(~J*J + W)*~J*(y_target - y); 


		cout << "iter " << i+1 << " pose diff = " << pose_diff << endl;

		kine_world.setJointState(q_start);
	}
	robot.move({q_start}, {5});
	robot.wait();
	//rai::wait();
	return q_start;
}

void circle(rai::KinematicWorld kine_world, arr q_home, RobotOperation robot) {
	arr Phi, PhiJ, J;
	arr y, y_target;
	arr q_start = 1.0*q_home;
	
	arr W;
	uint n = kine_world.getJointStateDimension();
	double w = rai::getParameter("w",1e-4);
	W.setDiag(w,n);  //W is equal the Id_n matrix times scalar w
	
	StringA null_motion_joints = {"right_s0", "right_s1", "right_e0", "right_e1", "right_w0", "right_w1"};
    	std::vector<arr> null_poses;
	for (int j=0; j<6; j++) {
		arr pose, Jtrash;
		kine_world.evalFeature(pose, Jtrash, FS_pose, {null_motion_joints(j)});
		null_poses.push_back(pose);
	}
	
	for (int i=0; i<10000; i++) {
		kine_world.getJointState(q_start);
		
		y_target = {.8, .0, 1.};
		y_target += .2 * arr({cos((double)i/20), 0, sin((double)i/20)});
		
		kine_world.evalFeature(y, J, FS_position, {"right_w2"});
		
		Phi.clear();
		PhiJ.clear();
		
		Phi.append(y-y_target);
		PhiJ.append( J );
		
		for(int n=0; n<6; n++)  {
			arr y_null, J_null;
			kine_world.evalFeature(y_null, J_null, FS_pose, {null_motion_joints(n)});
			Phi.append((y_null-null_poses[n])*0.1);
		    PhiJ.append( J_null*0.1 );
		}
		
		q_start -= 0.1*inverse(~PhiJ*PhiJ + W)*~PhiJ* Phi;
	    robot.move({q_start}, {4});
		robot.wait();

		cout << "iter " << i+1 << " pos diff = " << y-y_target << endl;

		kine_world.setJointState(q_start);
	}
}

void spline_use(){
	rai::KinematicWorld K;
	K.addFile("../../rai-robotModels/baxter/baxter.g");
	K.addObject("object", rai::ST_capsule, {.2, .05}, {1., 1., 0.}, -1., 0, {.8, .0, 1.});
	arr q_home = K.getJointState();

	RobotOperation B(K);
	cout <<"joint names: " <<B.getJointNames() <<endl;
	//B.move({q_zero}, {5.});
	//B.move({q_zero}, {5.}); //appends
	//B.wait();
	//rai::wait();

	//q_home(-1) = .1; //last joint set to .1: left gripper opens 10cm (or 20cm?)
	//B.move({q_home}, {4.});
	//B.wait();
    //circle(K, q_home, B);
	arr q_true = ik_algo(K, q_home, B);
	B.wait();
	printf("done returning to home\n");
	B.move({q_home}, {5});
	B.wait();
}


int main(int argc,char **argv){
	rai::initCmdLine(argc,argv);

	spline_use();
	return 0;
}


