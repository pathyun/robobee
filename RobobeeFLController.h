#pragma once

#include <memory>
#include <Eigen/Core>

#include "drake/examples/robobee/robobee_plant.h"
// #include "drake/examples/acrobot/gen/acrobot_input.h"
// #include "drake/examples/acrobot/gen/acrobot_params.h"
// #include "drake/examples/acrobot/gen/acrobot_state.h"
// #include "drake/math/wrap_to.h"
// #include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/framework/basic_vector.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/primitives/linear_system.h"
#include "drake/math/rotation_matrix.h"

/// @file The Spong acrobot swing-up controller implemented as a LeafSystem.
///   Spong, Mark W. "Swing up control of the acrobot." Robotics and Automation,
///   1994. Proceedings., 1994 IEEE International Conference on. IEEE, 1994.
///
/// Note that the Spong controller works well on the default set of parameters,
/// which Spong used in his paper. In contrast, it is difficult to find a
/// functional set of gains to stabilize the robot about its upright fixed
/// point using the parameters of the physical robot we have in lab.
///

namespace drake {
namespace examples {
namespace robobee {

using std::cout;
using std::endl;

template <typename T>
class RobobeeFLController : public systems::LeafSystem<T> {
 public:
  RobobeeFLController(double m_arg, const Eigen::Matrix3d& I_arg, const Eigen::MatrixXd& Mout, const Eigen::MatrixXd& Bout)
      : robobee_{}, robobee_context_(robobee_.CreateDefaultContext()),
      g_{9.81}, m_(m_arg), I_(I_arg), Mout_(Mout), Bout_(Bout)  {
    this->DeclareInputPort(systems::kVectorValued, kInputDimension);
    this->DeclareVectorOutputPort(systems::BasicVector<T>(kStateDimension),
                                &RobobeeFLController::CalcControl);
    /// Scalar-converting copy constructor.  See @ref system_scalar_conversion.
  
  
  }

  
  void CalcControl(const systems::Context<T>& context,
                         systems::BasicVector<T>* output) const {

    const VectorX<T> x = this->EvalVectorInput(context, 0)->get_value();

    const Vector4<T> u(1,0,0,0);

    // [0] Reference trajectory generation (example for circle and hovering at 0.3)
    T t = context.get_time();
    T x_f, y_f, z_f, dx_f, dy_f, ddx_f, ddy_f, dddx_f, dddy_f, ddddx_f, ddddy_f, radius, w_freq, T_period;

    T_period = 5.; // Period
    w_freq = 2*M_PI*1/T_period; // Rad freq
    radius = 0.5;
    x_f = radius*cos(w_freq*t);
    y_f = radius*sin(w_freq*t);
    dx_f = -radius*pow(w_freq,1)*sin(w_freq*t);
    dy_f = radius*pow(w_freq,1)*cos(w_freq*t);
    ddx_f = -radius*pow(w_freq,2)*cos(w_freq*t);
    ddy_f = -radius*pow(w_freq,2)*sin(w_freq*t);
    dddx_f = radius*pow(w_freq,3)*sin(w_freq*t);
    dddy_f = -radius*pow(w_freq,3)*cos(w_freq*t);
    ddddx_f = radius*pow(w_freq,4)*cos(w_freq*t);
    ddddy_f = radius*pow(w_freq,4)*sin(w_freq*t);

    z_f = 0.3; // Hovering height

    std::cout << "time: " << t << "\n";

    // [1] Useful Matrix and vectors

    T q0, q1, q2, q3;

    Vector3<T> e1(1,0,0);
    Vector3<T> e2(0,1,0);
    Vector3<T> e3(0,0,1);

    VectorX<T> q = Eigen::VectorXd::Zero(7);
    VectorX<T> qd = Eigen::VectorXd::Zero(6);
    
    q = x.template segment<7>(0);
    qd = x.template segment<6>(7);
    T xi1 = x(14);
    T xi2 = x(13);
    Vector3<T> v=qd.template segment<3>(0);
    Vector3<T> w=qd.template segment<3>(3);

    // Convert to Eigen quaternion
    Vector4<T> quat_vector = q.template segment<4>(3);
    q0=quat_vector(0);
    q1=quat_vector(1);
    q2=quat_vector(2);
    q3=quat_vector(3);

    quat_vector(3)=q0;
    quat_vector(0)=q1;
    quat_vector(1)=q2;
    quat_vector(2)=q3;

    const Eigen::Quaternion<T> quat(quat_vector);

    const drake::math::RotationMatrix<T> Rq(quat);
    
    MatrixX<T> Eq = Eigen::MatrixXd::Zero(3,4);

    Eq.block(0,0,1,4) =  Eigen::Vector4d(-1*q1,  q0, -1*q3,  q2).transpose();
    Eq.block(1,0,1,4) << Eigen::Vector4d(-1*q2,  q3,  q0, -1*q1).transpose();
    Eq.block(2,0,1,4) << Eigen::Vector4d(-1*q3, -1*q2,  q1,  q0).transpose();

    const Vector3<T> wIw = w.cross(I_ * w);            // Expressed in B
    
    const MatrixX<T> I_inv = I_.inverse();
    
    MatrixX<T> F1q = Eigen::MatrixXd::Zero(3,4);

    F1q.block(0,0,1,4) = Eigen::Vector4d( q2,  q3,  q0, q1).transpose();
    F1q.block(1,0,1,4) = Eigen::Vector4d(-q1, -q0,  q3, q2).transpose();
    F1q.block(2,0,1,4) = Eigen::Vector4d( q0, -q1, -q2, q3).transpose();
    
    cout <<"q:" << q << "\n";
    cout <<"qd:"<< qd << "\n";
    cout <<"v:"<< v << "\n";
    cout <<"w"<< w << "\n";
    cout <<"xi1:"<< xi1 << "\n";
    cout <<"xi2:"<< xi2 << "\n";
    
    
    Vector3<T> Rqe3 = Eigen::Vector3d::Zero();
    Matrix3<T> Rqe3_hat = Eigen::Matrix3d::Zero();
    Rqe3 = Rq.matrix()*e3;
    Rqe3_hat.block(0,0,1,3) = Eigen::Vector3d(       0, -Rqe3(2),  Rqe3(1)).transpose();
    Rqe3_hat.block(1,0,1,3) = Eigen::Vector3d( Rqe3(2),        0, -Rqe3(0)).transpose();
    Rqe3_hat.block(2,0,1,3) = Eigen::Vector3d(-Rqe3(1),  Rqe3(0),        0).transpose();

    Vector3<T> Rqe1 = Eigen::Vector3d::Zero();
    Matrix3<T> Rqe1_hat = Eigen::Matrix3d::Zero();
    Rqe1 = Rq.matrix()*e1;
    Rqe1_hat.block(0,0,1,3) = Eigen::Vector3d(       0, -Rqe1(2),  Rqe1(1)).transpose();
    Rqe1_hat.block(1,0,1,3) = Eigen::Vector3d( Rqe1(2),        0, -Rqe1(0)).transpose();
    Rqe1_hat.block(2,0,1,3) = Eigen::Vector3d(-Rqe1(1),  Rqe1(0),        0).transpose();

    T Rqe1_x=e2.dot(Rqe1);
    T Rqe1_y=e1.dot(Rqe1);

    Matrix3<T> w_hat = Eigen::Matrix3d::Zero();
    w_hat.block(0,0,1,3) = Eigen::Vector3d(    0, -w(2),  w(1)).transpose();
    w_hat.block(1,0,1,3) = Eigen::Vector3d( w(2),     0, -w(0)).transpose();
    w_hat.block(2,0,1,3) = Eigen::Vector3d(-w(1),  w(0),     0).transpose();


    // #- Checking the derivation
    // cout << "\n Rq : \n"<< Rq.matrix() << "\n";
    // cout << "\n F1q : \n"<< F1q << "\n";
    // cout << "\n Eq : \n"<< Eq << "\n";
    // cout << "\n F1qEqT-(-Rqe3_hat) : \n"<< F1q*Eq.transpose()-(-Rqe3_hat) << "\n";
    // Vector3<T> Rqe3_cal = Eigen::Vector3d::Zero();
    // Rqe3_cal(0) = 2*(q3*q1+q0*q2);
    // Rqe3_cal(1) = 2*(q3*q2-q0*q1);
    // Rqe3_cal(2) = (q0*q0+q3*q3-q1*q1-q2*q2);

    // cout << "Rqe3 - Rqe3_cal: "<< Rqe3-Rqe3_cal;

    // # Four output
    T y1, y2, y3, y4;

    y1 = q(0)-x_f;
    y2 = q(1)-y_f;
    y3 = q(2)-z_f;
    y4 = atan2(Rqe1_x,Rqe1_y)-M_PI/8;
    
    Vector3<T>eta1 = Eigen::Vector3d::Zero();
    T eta5 =y4;

    eta1 = Eigen::Vector3d(y1,y2,y3);
    eta5 = y4;
    
    // cout <<"y4" << y4;
   
    // # First derivative of first three output and yaw output
    T dy1, dy2, dy3, dy4, x2y2, eta6;
    Vector3<T> eta2 = Eigen::Vector3d::Zero();
    eta2 = v - Eigen::Vector3d(dx_f,dy_f,0);
    dy1 = eta2(0);
    dy2 = eta2(1);
    dy3 = eta2(2);
    dy4 = 0;
    // cout << "eta2 :" << eta2;
    
    x2y2 = (pow(Rqe1_x,2)+pow(Rqe1_y,2));

    Vector3<T> eta6_temp = Eigen::Vector3d::Zero(); // eta6_temp = (ye2T-xe1T)/(x^2+y^2)
    eta6_temp = (Rqe1_y*e2.transpose()-Rqe1_x*e1.transpose())/x2y2;  
    // print("eta6_temp:", eta6_temp)
    eta6 = eta6_temp.dot(-Rqe1_hat*w) -dy4;
    // print("Rqe1_hat:", Rqe1_hat)

    // # Second derivative of first three output
    T ddy1, ddy2, ddy3;
    Vector3<T> eta3 = Eigen::Vector3d::Zero(); // eta6_temp = (ye2T-xe1T)/(x^2+y^2)
    eta3 = -g_*e3+Rqe3*xi1 - Eigen::Vector3d(ddx_f,ddy_f,0);
    ddy1 = eta3(0);
    ddy2 = eta3(1);
    ddy3 = eta3(2);

    // # Third derivative of first three output
    T dddy1, dddy2, dddy3;
    Vector3<T> eta4 = Eigen::Vector3d::Zero(); 
    eta4 = Rqe3*xi2+F1q*Eq.transpose()*w*xi1 - Eigen::Vector3d(dddx_f,dddy_f,0);
    dddy1 = eta4(0);
    dddy2 = eta4(1);
    dddy3 = eta4(2);

    // # Fourth derivative of first three output
    Vector3<T> B_qw_temp = Eigen::Vector3d::Zero();
    Vector3<T> B_qw = Eigen::Vector3d::Zero();
    
    B_qw_temp = xi1*(-w_hat*Rqe3_hat*w+Rqe3_hat*I_inv*wIw); // # np.dot(I_inv,wIw)*xi1-2*w*xi2
    B_qw      = B_qw_temp+xi2*(-2*Rqe3_hat*w) - Eigen::Vector3d(ddddx_f,ddddy_f,0); //   #np.dot(Rqe3_hat,B_qw_temp)

    // # Second derivative of yaw output
    T dRqe1_x, dRqe1_y, alpha1, B_yaw;

    dRqe1_x = e2.dot(-Rqe1_hat*w); // # \dot{x}
    dRqe1_y = e1.dot(-Rqe1_hat*w); // # \dot{y}

    alpha1 = 2*(Rqe1_x*dRqe1_x+Rqe1_y*dRqe1_y)/x2y2; //# (2xdx +2ydy)/(x^2+y^2)
    // # alpha2 = math.pow(dRqe1_y,2)-math.pow(dRqe1_x,2)

    Vector3<T> B_yaw_temp3 = Eigen::Vector3d::Zero();
    
    B_yaw_temp3 = alpha1*Rqe1_hat*w+Rqe1_hat*I_inv*wIw-w_hat*Rqe1_hat*w;

    B_yaw = eta6_temp.dot(B_yaw_temp3); // # +alpha2 :Could be an error in math.

    Vector3<T> g_yaw = Eigen::Vector3d::Zero();
    g_yaw = -eta6_temp.transpose()*Rqe1_hat*I_inv;

    // print("g_yaw:", g_yaw)
    // # Decoupling matrix A(x)\in\mathbb{R}^4
    MatrixX<T> A_fl = Eigen::MatrixXd::Zero(4,4);
    A_fl.block(0,0,3,1) = Rqe3;
    A_fl.block(0,1,3,3) = -Rqe3_hat*I_inv*xi1;
    A_fl.block(3,1,1,3) = g_yaw.transpose();

    T A_fl_det;
    MatrixX<T> A_fl_inv = Eigen::MatrixXd::Zero(4,4);
    
    A_fl_inv = A_fl.inverse();
    A_fl_det = A_fl.determinant();
    //# print("I_inv:", I_inv)
    // cout << "\n Rqe3 : \n" << Rqe3 ;
    // cout << "\n -Rqe3_hat*I_inv*xi1 : \n" << -Rqe3_hat*I_inv*xi1 ;
    // cout << "\n g_yaw : \n" << g_yaw ;
    cout << "\n A_fl: \n"<< A_fl << "\n";
    cout << "\n A_fl_det: \n"<< A_fl_det << "\n";



    // # Output dyamics
    VectorX<T> eta = Eigen::VectorXd::Zero(14);
    eta << eta1, eta2, eta3, eta5, eta4, eta6;

    // # Full feedback controller
    VectorX<T> U_temp = Eigen::VectorXd::Zero(4);

    U_temp <<  B_qw, B_yaw;

    // cout << "\n B_qw: \n" << B_qw;
    // cout << "\n B_yaw: \n" << B_yaw;
    // cout << "\n U_temp: \n" << U_temp;
    // cout << "\n eta_norm: \n" << eta.dot(Mout_*eta) << "\n";
    
    VectorX<T> mu = Eigen::VectorXd::Zero(4);
    MatrixX<T> k = Eigen::MatrixXd::Zero(4,14);
                
    k = Bout_.transpose()*Mout_;
    // cout<< "\n Bout_ : \n" << Bout_;
    // cout<< "\n Mout_ : \n" << Mout_;
    // cout<< "\n k : \n" << k;


    mu = -k*eta;

    VectorX<T> v_temp = Eigen::VectorXd::Zero(4);
    VectorX<T> U_fl = Eigen::VectorXd::Zero(4);
    v_temp=-U_temp+mu;

    cout << "\n v_temp : \n" << v_temp ;
    
    U_fl = A_fl_inv*v_temp; //       # Feedback controller
    // U_fl =A_fl.ldlt().solve(v_temp);

    cout << "\n Feedback Controller : \n" << U_fl << "\n";
    output->set_value(u);


  }

 private:
  RobobeePlant<T> robobee_;
  // The implementation above is (and must remain) careful to not store hidden
  // state in here.  This is only used to avoid runtime allocations.

  static constexpr int kStateDimension{4};
  static constexpr int kInputDimension{15};

  const std::unique_ptr<systems::Context<T>> robobee_context_;

  double g_;           // Gravitational acceleration (m/s^2).
  double m_;           // Mass of the robot (kg).
  Eigen::Matrix3d I_;  // Moment of Inertia about the Center of Mass
  Eigen::MatrixXd Mout_;  // Moment of Inertia about the Center of Mass
  Eigen::MatrixXd Bout_;  // Moment of Inertia about the Center of Mass
};

}  // namespace acrobot
}  // namespace examples
}  // namespace drake