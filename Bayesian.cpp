//
// Created by wjh on 23-12-18.
//
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

#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif

#ifndef EIGEN_VECTORIZE_SSE4_2
#define EIGEN_VECTORIZE_SSE4_2
#endif

#pragma GCC optimize(3)

using namespace std;

enum ACTION {
    UP = 0, DOWN, LEFT, RIGHT
};


int main() {
//    Eigen::setNbThreads(10)
    omp_set_num_threads(6);
    double instant_reward = -1;

    const int total_state = 16;

    // 观测的似然
    Eigen::Matrix2Xd p_obs_state(2, total_state);
    p_obs_state.setConstant(0.5);
    p_obs_state(0, 14) = 1;
    p_obs_state(1, 14) = 0;
    p_obs_state(0, 15) = 0;
    p_obs_state(1, 15) = 1;

    // 状态转移
    vector<Eigen::Matrix<double, total_state, total_state>> trans_vec;
    Eigen::Matrix<double, total_state, total_state> transition(total_state, total_state);
    transition.setConstant(0);
    trans_vec.push_back(transition);
    trans_vec.push_back(transition);
    trans_vec.push_back(transition);
    trans_vec.push_back(transition);

    for (int i = 0; i < total_state; i++) {
        switch (i / 2) {
            case 0:
                trans_vec[UP](i, i) = 1;
                trans_vec[DOWN](i, i) = 1;
                trans_vec[LEFT](i, i) = 1;
                trans_vec[RIGHT](i, i) = 1;
                break;
            case 1:
                trans_vec[UP](i, i) = 1;
                trans_vec[DOWN](i, i) = 1;
                trans_vec[LEFT](i, i) = 1;
                trans_vec[RIGHT](i, i) = 1;
                break;
            case 2:
                trans_vec[UP](i, i - 4) = 1;
                trans_vec[DOWN](i, i) = 1;
                trans_vec[LEFT](i, i) = 1;
                trans_vec[RIGHT](i, i + 2) = 1;
                break;
            case 3:
                trans_vec[UP](i, i) = 1;
                trans_vec[DOWN](i, i + 4) = 1;
                trans_vec[LEFT](i, i - 2) = 1;
                trans_vec[RIGHT](i, i + 2) = 1;
                break;
            case 4:
                trans_vec[UP](i, i - 6) = 1;
                trans_vec[DOWN](i, i) = 1;
                trans_vec[LEFT](i, i - 2) = 1;
                trans_vec[RIGHT](i, i) = 1;
                break;
            case 5:
                trans_vec[UP](i, i - 4) = 1;
                trans_vec[DOWN](i, i + 2) = 1;
                trans_vec[LEFT](i, i) = 1;
                trans_vec[RIGHT](i, i) = 1;
                break;
            case 6:
                trans_vec[UP](i, i - 2) = 1;
                trans_vec[DOWN](i, i) = 1;
                trans_vec[LEFT](i, i) = 1;
                trans_vec[RIGHT](i, i + 2) = 1;
                break;
            case 7:
                trans_vec[UP](i, i) = 1;
                trans_vec[DOWN](i, i) = 1;
                trans_vec[LEFT](i, i - 2) = 1;
                trans_vec[RIGHT](i, i) = 1;
                break;
        }
    }

//    cout << "hello";
}
