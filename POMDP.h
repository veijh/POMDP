//
// Created by wjh on 23-12-29.
//

#ifndef CPP_TEST_POMDP_H
#define CPP_TEST_POMDP_H

#include "iostream"
#include <iomanip>
#include "vector"
#include "algorithm"
#include "unordered_map"
#include "map"
#include "omp.h"
#include "Eigen/Eigen"
#include "Eigen/Dense"
#include "set"
#include "cstdlib"

using namespace std;

class POMDP{
private:
    // dimension of state_dim space: S
    int state_dim;
    // dimension of action space: A
    int act_dim;
    // dimension of observation space: O
    int obs_dim;
    // state_dim transition probabilities taking specific action: AxSxS
    vector<Eigen::MatrixXd> trans_vec;
    // immediate rewards: SxA
    Eigen::MatrixXd rwd_s_a;
    // observation probabilities: OxS
    Eigen::MatrixXd p_obs_in_s;
    // belief points: (1+S)xN
    Eigen::MatrixXd belief_points;
    // each point corresponds with an alpha vector: Nx(1+S)
    Eigen::MatrixXd alpha_vector;
public:
    POMDP(const vector<Eigen::MatrixXd> &transition, const Eigen::MatrixXd &r_s_a, const Eigen::MatrixXd &p_o_s);
    void PBVI(Eigen::MatrixXd _belief_points, int horizon_len);
    int select_action(Eigen::VectorXd _belief_state);
};

#endif //CPP_TEST_POMDP_H
