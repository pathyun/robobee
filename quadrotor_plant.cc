#include "drake/examples/robobee/quadrotor_plant.h"

#include <memory>

#include "drake/common/default_scalars.h"
#include "drake/math/gradient.h"
#include "drake/math/rotation_matrix.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/util/drakeGeometryUtil.h"

using Eigen::Matrix3d;

namespace drake {
namespace examples {
namespace robobee {

namespace {
/*  Matrix3d default_moment_of_inertia() {
  return (Eigen::Matrix3d() <<  // BR
          0.0023, 0, 0,  // BR
          0, 0.0023, 0,  // BR
          0, 0, 0.0040).finished();
}
}  // namespace

template <typename T>
QuadrotorPlant<T>::QuadrotorPlant()
    : QuadrotorPlant(0.5,    // m (kg)
                     0.175,  // L (m)
                     default_moment_of_inertia(),
                     1.0,    // kF
                     0.0245  // kM
                     ) {}
*/                     

Matrix3d default_moment_of_inertia() {
  return (Eigen::Matrix3d() <<  // BR
          0.0000000018746, 0, 0,  // BR
          0, 0.00000000197824, 0,  // BR
          0, 0, 0.000000000637).finished();
}
}  // namespace

template <typename T>
QuadrotorPlant<T>::QuadrotorPlant()
    : QuadrotorPlant(0.000080,    // m (kg)
                     0.175,  // L (m)
                     default_moment_of_inertia(),
                     1.0,    // kF
                     0.0245  // kM
                     ) {}

template <typename T>
QuadrotorPlant<T>::QuadrotorPlant(double m_arg, double L_arg,
                                  const Matrix3d& I_arg, double kF_arg,
                                  double kM_arg)
    : systems::LeafSystem<T>(
          systems::SystemTypeTag<robobee::QuadrotorPlant>{}),
      g_{9.81}, m_(m_arg), L_(L_arg), kF_(kF_arg), kM_(kM_arg), I_(I_arg) {
  this->DeclareInputPort(systems::kVectorValued, kInputDimension);
  this->DeclareContinuousState(kStateDimension);
  this->DeclareVectorOutputPort(systems::BasicVector<T>(kStateDimension),
                                &QuadrotorPlant::CopyStateOut);
}

template <typename T>
template <typename U>
QuadrotorPlant<T>:: QuadrotorPlant(const QuadrotorPlant<U>& other)
    : QuadrotorPlant<T>(other.m_, other.L_, other.I_, other.kF_, other.kM_) {}

template <typename T>
QuadrotorPlant<T>::~QuadrotorPlant() {}

template <typename T>
void QuadrotorPlant<T>::CopyStateOut(const systems::Context<T> &context,
                                     systems::BasicVector<T> *output) const {
  output->set_value(
      context.get_continuous_state_vector().CopyToVector());
}

template <typename T>
void QuadrotorPlant<T>::DoCalcTimeDerivatives(
    const systems::Context<T> &context,
    systems::ContinuousState<T> *derivatives) const {
  // Get the input value characterizing each of the 4 rotor's aerodynamics.
  const Vector4<T> u = this->EvalVectorInput(context, 0)->get_value();

  // For each rotor, calculate the Bz measure of its aerodynamic force on B.
  // Note: B is the quadrotor body and Bz is parallel to each rotor's spin axis.
  // u= [F_T, tau_1, tau2, tau3] : Robobee input Thrust and torque in the body frame
  const Vector4<T> uF_Bz = u;

  // Compute the net aerodynamic force on B (from the 4 rotors), expressed in B.
  const Vector3<T> Faero_B(0, 0, m_*uF_Bz(0));

  // Compute the Bx and By measures of the moment on B about Bcm (B's center of
  // mass) from the 4 rotor forces.  These moments arise from the cross product
  // of a position vector with an aerodynamic force at the center of each rotor.
  // For example, the moment of the aerodynamic forces on rotor 0 about Bcm
  // results from Cross( L_* Bx, uF_Bz(0) * Bz ) = -L_ * uF_Bz(0) * By.
  const T Mx = uF_Bz(1);
  const T My = uF_Bz(2);

  // For rotors 0 and 2, get the Bz measure of its aerodynamic torque on B.
  // For rotors 1 and 3, get the -Bz measure of its aerodynamic torque on B.
  // Sum the net Bz measure of the aerodynamic torque on B.
  // Note: Rotors 0 and 2 rotate one way and rotors 1 and 3 rotate the other.
  const Vector4<T> uTau_Bz = u;
  const T Mz = 0;//uTau_Bz(3);

  // Form the net moment on B about Bcm, expressed in B. The net moment accounts
  // for all contact and distance forces (aerodynamic and gravity forces) on B.
  // Note: Since the net moment on B is about Bcm, gravity does not contribute.
  // const Vector3<T> Tau_B(Mx, My, Mz);
  const Vector3<T> Tau_B(Mx, My, Mz);
  //std::cout << "Tau_B: " << Tau_B(2) <<"\n";
  // Calculate local celestial body's (Earth's) gravity force on B, expressed in
  // the Newtonian frame N (a.k.a the inertial or World frame).
  const Vector3<T> Fgravity_N(0, 0, -m_ * g_);

  // Extract roll-pitch-yaw angles (rpy) and their time-derivatives (rpyDt).
  VectorX<T> state = context.get_continuous_state_vector().CopyToVector();
  const drake::math::RollPitchYaw<T> rpy(state.template segment<3>(3));
  const Vector3<T> rpyDt = state.template segment<3>(9);

  // Convert roll-pitch-yaw (rpy) orientation to the R_NB rotation matrix.
  const drake::math::RotationMatrix<T> R_NB(rpy);

  // Calculate the net force on B, expressed in N.  Use Newton's law to
  // calculate a_NBcm_N (acceleration of B's center of mass, expressed in N).
  const Vector3<T> Fnet_N = Fgravity_N + R_NB * Faero_B;
  const Vector3<T> xyzDDt = Fnet_N / m_;  // Equal to a_NBcm_N.

  // Use rpy and rpyDt to calculate B's angular velocity in N, expressed in B.
  const Vector3<T> w_BN_B = rpy.CalcAngularVelocityInChildFromRpyDt(rpyDt);

  // To compute α (B's angular acceleration in N) due to the net moment 𝛕 on B,
  // rearrange Euler rigid body equation  𝛕 = I α + ω × (I ω)  and solve for α.
  const Vector3<T> wIw = w_BN_B.cross(I_ * w_BN_B);            // Expressed in B
  const Vector3<T> alpha_NB_B = I_.ldlt().solve(Tau_B - wIw);  // Expressed in B
  const Vector3<T> alpha_NB_N = R_NB * alpha_NB_B;             // Expressed in N

  // Calculate the 2nd time-derivative of rpy.
  const Vector3<T> rpyDDt =
      rpy.CalcRpyDDtFromRpyDtAndAngularAccelInParent(rpyDt, alpha_NB_N);

  // Recomposing the derivatives vector.
  VectorX<T> xDt(12);
  xDt << state.template tail<6>(), xyzDDt, rpyDDt;
  derivatives->SetFromVector(xDt);
  //t << "\n";
  //std::cout << Faero_B(2) << "\n";
  std::cout << "Time : "<< context.get_time()<< "\n";

  std::cout << "Thrust to weight ratio : "<<Faero_B(2)/(-1*Fgravity_N(2)) << "\n";

  std::cout << "z Position : "<< state(2) << "\n";
  std::cout << "Thrust (mN/s^2) : "<< Faero_B(2)*1000 << "\n";
  std::cout << "Accellaration (cm/s^2) : "<< xyzDDt(2)*100 << "\n";
  std::cout << "r (rad): "<< state(3) << "\n";
  std::cout << "p (rad): "<< state(4) << "\n";
  std::cout << "y (rad): "<< state(5) << "\n";
  std::cout << "Tau_x (mNmm): "<< u(1)*1000000 << "\n";
  std::cout << "Tau_y (mNmm): "<< u(2)*1000000 << "\n";
  std::cout << "Tau_z (mNmm): "<< u(3)*1000000 << "\n";

  std::cout << "=================================\n";
}

// Declare storage for our constants.
template <typename T>
constexpr int QuadrotorPlant<T>::kStateDimension;
template <typename T>
constexpr int QuadrotorPlant<T>::kInputDimension;

std::unique_ptr<systems::AffineSystem<double>> StabilizingLQRController(
    const QuadrotorPlant<double>* quadrotor_plant,
    Eigen::Vector3d nominal_position) {
  auto quad_context_goal = quadrotor_plant->CreateDefaultContext();

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(12);
  x0.topRows(3) = nominal_position;


  // Nominal input corresponds to a hover.
  Eigen::VectorXd u0 = Eigen::VectorXd::Zero(4);
  u0(0)= quadrotor_plant->g();
  
//std::cout << "Nominal input: " << u0(0) << u0(1) << u0(2) << u0(3); 

  quad_context_goal->FixInputPort(0, u0);
  quadrotor_plant->set_state(quad_context_goal.get(), x0);

  // Setup LQR cost matrices (penalize position error 10x more than velocity
  // error).
  Eigen::MatrixXd Q = 0.01*Eigen::MatrixXd::Identity(12, 12);
  Q.topLeftCorner<6, 6>() = 100000 * Eigen::MatrixXd::Identity(6, 6);
  Q(2,2)=100000000000;
  std::cout << Q;
  Eigen::Matrix4d R = 100000*Eigen::Matrix4d::Identity();

  return systems::controllers::LinearQuadraticRegulator(
      *quadrotor_plant, *quad_context_goal, Q, R);
}

}  // namespace quadrotor
}  // namespace examples
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::examples::robobee::QuadrotorPlant)
