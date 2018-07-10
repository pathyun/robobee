// Generates a swing-up trajectory for robobee and displays the trajectory
// in DrakeVisualizer. Trajectory generation code is based on
// pendulum_swing_up.cc.

#include <iostream>
#include <memory>

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

  auto robobee = builder.AddSystem<RobobeePlant<double>>();
  robobee->set_name("robobee");
  //RobobeePlant<double> robobee;
  auto context = robobee->CreateDefaultContext();

  const int kNumTimeSamples = 21;
  const double kMinimumTimeStep = 0.2;
  const double kMaximumTimeStep = 0.5;
  systems::trajectory_optimization::DirectCollocation dircol(
      robobee, *context, kNumTimeSamples, kMinimumTimeStep,
      kMaximumTimeStep);

  dircol.AddEqualTimeIntervalsConstraints();

  // 
  //const double kTorqueLimit = 8;
  auto u = dircol.input();
  //dircol.AddConstraintToAllKnotPoints(-kTorqueLimit <= u(0));
  //dircol.AddConstraintToAllKnotPoints(u(0) <= kTorqueLimit);

  Eigen::VectorXd x0=Eigen::VectorXd::Zero(12);

  //x0(9)=1; // w1=1;
  //x0(10)=1;// w2=1;
  //x0(11)=1;// w3=1;

  Eigen::VectorXd xG=Eigen::VectorXd::Zero(12);

  xG(2)=0.1;
//  xG(9)=0; // w1=1;
//  xG(10)=0;// w2=1;
//  xG(11)=0;// w3=1;
  

 // const Eigen::VectorXd xG(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
//  dircol.AddLinearConstraint(dircol.initial_state() == x0);
//  dircol.AddLinearConstraint(dircol.final_state() == xG);
  dircol.AddLinearConstraint(dircol.initial_state() == x0);
  dircol.AddLinearConstraint(dircol.final_state() == xG);

  //const double R = 10;  // Cost on input "effort".
  Eigen::Matrix4d R = 20*Eigen::Matrix4d::Identity();

  dircol.AddRunningCost( (u.transpose()*R) * u);

  const double timespan_init = 10;
  auto traj_init_x =
      PiecewisePolynomialType::FirstOrderHold({0, timespan_init}, {x0, xG});
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
      multibody::joints::kRollPitchYaw, tree.get());
  
  auto visualizer =
      builder.AddSystem<drake::systems::DrakeVisualizer>(*tree, &lcm);
  visualizer->set_name("visualizer");
  builder.Connect(robobee->get_output_port(0), visualizer->get_input_port(0));

  auto publisher = builder.AddSystem<systems::DrakeVisualizer>(*tree, &lcm);

  // By default, the simulator triggers a publish event at the end of each time
  // step of the integrator. However, since this system is only meant for
  // playback, there is no continuous state and the integrator does not even get
  // called. Therefore, we explicitly set the publish frequency for the
  // visualizer.
  publisher->set_publish_period(1.0 / 60.0);

  builder.Connect(state_source->get_output_port(),
                  publisher->get_input_port(0));

  auto diagram = builder.Build();

  systems::Simulator<double> simulator(*diagram);

  simulator.get_mutable_context()
        .get_mutable_continuous_state_vector()
        .SetFromVector(x0);

  simulator.set_target_realtime_rate(FLAGS_realtime_factor);
  simulator.Initialize();
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
