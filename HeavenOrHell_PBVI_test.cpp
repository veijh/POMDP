//
// Created by wjh on 23-12-18.
//
#include "iostream"
#include <iomanip>
#include "vector"
#include "omp.h"
#include "Eigen/Eigen"
#include "Eigen/Dense"
#include "set"
#include "POMDP.h"

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

class INDEX_MAP {
private:
    int row, col, world, obs;
public:
    INDEX_MAP(int _row, int _col, int _world, int _obs) : row(_row), col(_col), world(_world), obs(_obs) {};

    int get_index(int i, int j, int _world) {
        return _world + world * j + world * col * i;
    }
};

int main() {
//    Eigen::setNbThreads(10)
    omp_set_num_threads(6);
    double instant_reward = -1;
    INDEX_MAP state_index(5, 4, 2, 2);

    const int total_state = 16;

    // 观测的似然
    Eigen::MatrixXd p_obs_state(3, total_state);
    p_obs_state.setConstant(0);
    p_obs_state.row(2).setConstant(1);
    p_obs_state(0, 14) = 1;
    p_obs_state(1, 14) = 0;
    p_obs_state(2, 14) = 0;
    p_obs_state(0, 15) = 0;
    p_obs_state(1, 15) = 1;
    p_obs_state(2, 15) = 0;

    // 状态转移
    vector<Eigen::MatrixXd> trans_vec;
    Eigen::MatrixXd transition(total_state, total_state);
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

    // 奖励函数
    // reward可以写成R(s,a)矩阵
    Eigen::MatrixXd reward(total_state, 4);
    reward.setConstant(-1);
    reward(4, UP) = 100;
    reward(5, UP) = -100;
    reward(8, UP) = -100;
    reward(9, UP) = 100;
    reward.row(0).setConstant(0);
    reward.row(1).setConstant(0);
    reward.row(2).setConstant(0);
    reward.row(3).setConstant(0);

    // PBVI的核心
    const int point_num = 3*total_state/2;
    Eigen::MatrixXd possible_state(2,3);
    possible_state << 1, 0, 0.5,
            0, 1, 0.5;
    int active_num = 0;
    // 信念点的集合 N x point_num
    Eigen::MatrixXd belief_point(total_state, point_num);
    belief_point.setConstant(0);
    for(int i = 0; i < total_state/2; i++){
        belief_point.block(2*i,3*i,2,3) = possible_state;
    }

    // 初始化与求解
    POMDP PBVI(trans_vec, reward, p_obs_state);
    PBVI.PBVI(belief_point, 10);

    vector<vector<int>> policy;
    for(int s = 0; s < total_state/2; s++){
        Eigen::VectorXd belief(total_state);
        belief.setConstant(0);
        belief(2*s) = 0.5;
        belief(2*s+1) = 0.5;
        vector<int> best_actions = PBVI.select_action(belief);
        policy.push_back(best_actions);
    }

    int index = 0;
    for(auto i:policy){
        switch (index) {
            case 1:
            case 5:
            case 6:
                cout << setw(5) << " ";
                break;
            default:
                break;
        }
        string output;
        for(auto j:i){
            switch (j) {
                case UP:
                    output += "^";
                    break;
                case LEFT:
                    output += "<";
                    break;
                case RIGHT:
                    output += ">";
                    break;
                case DOWN:
                    output += "v";
                    break;
            }
        }
        cout << setw(5) << output;
        switch (index) {
            case 1:
            case 4:
            case 5:
            case 7:
                cout << endl;
                break;
            default:
                break;
        }
        index++;
    }

}
