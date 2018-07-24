// Generates a swing-up trajectory for robobee and displays the trajectory
// in DrakeVisualizer. Trajectory generation code is based on
// pendulum_swing_up.cc.

#include <iostream>
#include <memory>
#define _USE_MATH_DEFINES
#include <math.h>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/examples/robobee/robobee_plant.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/joints/floating_base_types.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/rigid_body_plant/drake_visualizer.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/systems/trajectory_optimization/direct_collocation.h"
#include "drake/multibody/rigid_body_plant/rigid_body_plant.h"
#include "drake/multibody/rigid_body_tree.h"
#include "drake/multibody/rigid_body_tree_construction.h"

using drake::solvers::SolutionResult;


namespace drake {
namespace examples {
namespace robobee {

typedef trajectories::PiecewisePolynomial<double> PiecewisePolynomialType;

namespace {
DEFINE_double(realtime_factor, 1.0,
              "Playback speed.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");

int do_main() {
  systems::DiagramBuilder<double> builder;

  // auto robobee = builder.AddSystem<RobobeePlant<double>>();
  // robobee->set_name("robobee");
  RobobeePlant<double> robobee;
  auto context = robobee.CreateDefaultContext();

  // auto context = robobee->CreateDefaultContext();

  const int kNumTimeSamples = 20;
  const double kMinimumTimeStep = 0.02;
  const double kMaximumTimeStep = 0.5;
  systems::trajectory_optimization::DirectCollocation dircol(
      &robobee, *context, kNumTimeSamples, kMinimumTimeStep,
      kMaximumTimeStep);

  dircol.AddEqualTimeIntervalsConstraints();

  // 
  // const double kTorqueLimit = 8000;

  std::cout << "State size: " << robobee.get_num_states() << "\n";

  std::cout << "dir state size" << context->get_continuous_state().size();
  auto u = dircol.input();
  // dircol.AddConstraintToAllKnotPoints(-kTorqueLimit <= u(0));
  // dircol.AddConstraintToAllKnotPoints(u(0) <= kTorqueLimit);

// [0] Initial configuration in SE(3)
  Eigen::VectorXd x0=Eigen::VectorXd::Zero(13);

  // Position r
  x0(0) = 0.;
  x0(1) = 0.;
  x0(2) = 0.;

  // Orientation q (quaternion)
  double theta0 = M_PI/4;  // angle of otation
  Eigen::Vector3d v0_q=Eigen::Vector3d:: Zero(3); 
  v0_q(0) = 1.;
  v0_q(1) = 0.;
  v0_q(2) = 1.;
  
  Eigen::VectorXd q = Eigen::VectorXd::Zero(4);
  q(0)= cos(theta0/2);
  double v0_norm; 
  v0_norm = sqrt(v0_q.transpose()*v0_q);

  Eigen::VectorXd v0_normalized = Eigen::VectorXd::Zero(3);
  v0_normalized = v0_q/v0_norm;

  q(1)= sin(theta0/2)*v0_normalized(0);
  q(2)= sin(theta0/2)*v0_normalized(1);
  q(3)= sin(theta0/2)*v0_normalized(2);

  x0(3) =q(0);
  x0(4) =q(1);
  x0(5) =q(2);
  x0(6) =q(3);

  // Angular velocity w
  Eigen::Vector3d w0 = Eigen::Vector3d::Zero(3);
  w0(0)= -0.1;
  w0(1)=  0.;
  w0(2)=  0.1 ;
  
  x0(10)=w0(0); // w1=1;
  x0(11)=w0(1);// w2=1;
  x0(12)=w0(2);// w3=1;
  std::cout << "Intial condition x0:" << x0 <<"\n";

// [1] Final configuration in SE(3)
  Eigen::VectorXd xf=Eigen::VectorXd::Zero(13);
  
  // Position r
  xf(0) = 0.;
  xf(1) = 0.;
  xf(2) = 0.3;

  // Orientation q (quaternion)
  double thetaf = M_PI/1;  // angle of otation
  Eigen::Vector3d vf_q=Eigen::Vector3d:: Zero(3); 
  vf_q(0) = 1.;
  vf_q(1) = 0.;
  vf_q(2) = -0.;
  
  Eigen::VectorXd qf = Eigen::VectorXd::Zero(4);
  qf(0)= cos(thetaf/2);
  double vf_norm; 
  vf_norm = sqrt(vf_q.transpose()*vf_q);

  Eigen::VectorXd vf_normalized = Eigen::VectorXd::Zero(3);
  vf_normalized = vf_q/vf_norm;

  qf(1)= sin(thetaf/2)*vf_normalized(0);
  qf(2)= sin(thetaf/2)*vf_normalized(1);
  qf(3)= sin(thetaf/2)*vf_normalized(2);
  // std::cout << "qf:" << qf <<"\n";

  xf(3) =qf(0);
  xf(4) =qf(1);
  xf(5) =qf(2);
  xf(6) =qf(3);

  // Angular velocity w
  Eigen::Vector3d wf = Eigen::Vector3d::Zero(3);
  wf(0)= -0.;
  wf(1)=  0.;
  wf(2)=  0. ;
  
  xf(10)=wf(0); // w1=1;
  xf(11)=wf(1);// w2=1;
  xf(12)=wf(2);// w3=1;


  std::cout << "Final condition xf:" << xf <<"\n";
 
 // const Eigen::VectorXd xG(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
//  dircol.AddLinearConstraint(dircol.initial_state() == x0);
//  dircol.AddLinearConstraint(dircol.final_state() == xG);
  dircol.AddBoundingBoxConstraint(x0, x0,
                                dircol.initial_state());
  dircol.AddBoundingBoxConstraint(xf, xf,
                                dircol.final_state());
  // dircol.AddLinearConstraint(dircol.initial_state() == x0);
  // dircol.AddLinearConstraint(dircol.final_state() == xG);

  //const double R = 10;  // Cost on input "effort".
  Eigen::Matrix4d R = 20*Eigen::Matrix4d::Identity();

  dircol.AddRunningCost( (u.transpose()*R) * u);

  const double timespan_init = 10;
  auto traj_init_x =
      PiecewisePolynomialType::FirstOrderHold({0, timespan_init}, {x0, x0});
  dircol.SetInitialTrajectory(PiecewisePolynomialType(), traj_init_x);


  SolutionResult result = dircol.Solve();

  if (result != SolutionResult::kSolutionFound) {
    std::cerr << "No solution found.\n";
    return 1;
  }  

  const trajectories::PiecewisePolynomial<double> pp_xtraj =
      dircol.ReconstructStateTrajectory();
  auto state_source = builder.AddSystem<systems::TrajectorySource>(pp_xtraj);

  lcm::DrakeLcm lcm;
  auto tree = std::make_unique<RigidBodyTree<double>>();
  parsers::urdf::AddModelInstanceFromUrdfFileToWorld(
      FindResourceOrThrow("drake/examples/robobee/robobee.urdf"),
      multibody::joints::kQuaternion, tree.get());
  
  auto publisher = builder.AddSystem<systems::DrakeVisualizer>(*tree, &lcm);

  // By default, the simulator triggers a publish event at the end of each time
  // step of the integrator. However, since this system is only meant for
  // playback, there is no continuous state and the integrator does not even get
  // called. Therefore, we explicitly set the publish frequency for the
  // visualizer.
  
  publisher->set_publish_period(1.0 / 12000.0);

  builder.Connect(state_source->get_output_port(),
                  publisher->get_input_port(0));

  auto diagram = builder.Build();

  systems::Simulator<double> simulator(*diagram);

  simulator.set_target_realtime_rate(FLAGS_realtime_factor);
  simulator.Initialize();
  simulator.StepTo(pp_xtraj.end_time());

  std::cout << "Ending time: "<< pp_xtraj.end_time() << "\n";

  Eigen::VectorXd pp_temp = pp_xtraj.value(pp_xtraj.end_time());
  std::cout << "pp_xtraj: " << pp_temp <<"\n";


  simulator.StepTo(pp_xtraj.end_time());

  
  return 0;
}

}  // namespace
}  // namespace robobee
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::robobee::do_main();
}
