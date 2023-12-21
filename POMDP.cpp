//
// Created by wjh on 23-12-18.
//
#include "iostream"
#include <iomanip>
#include "vector"
#include "unordered_map"
#include "map"
#include "Eigen/Eigen"
#include "Eigen/Dense"
#include "iostream"
#include <iomanip>
#include "vector"
#include "unordered_map"
#include "map"
#include "Eigen/Eigen"
#include "Eigen/Dense"

#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif

#ifndef EIGEN_VECTORIZE_SSE4_2
#define EIGEN_VECTORIZE_SSE4_2
#endif

#pragma GCC optimize(2)

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
    double instant_reward = -1;
    INDEX_MAP state_index(5, 4, 2, 2);
    Eigen::Matrix2Xd p_obs_state(2, 5 * 4 * 2);
    p_obs_state.setConstant(0.5);
    p_obs_state(0, state_index.get_index(0, 1, 0)) = 1;
    p_obs_state(0, state_index.get_index(0, 1, 1)) = 0;
    p_obs_state(1, state_index.get_index(0, 1, 0)) = 0;
    p_obs_state(1, state_index.get_index(0, 1, 1)) = 1;

    p_obs_state(0, state_index.get_index(2, 1, 0)) = 1;
    p_obs_state(0, state_index.get_index(2, 1, 1)) = 0;
    p_obs_state(1, state_index.get_index(2, 1, 0)) = 0;
    p_obs_state(1, state_index.get_index(2, 1, 1)) = 1;

    p_obs_state(0, state_index.get_index(0, 3, 0)) = 1;
    p_obs_state(0, state_index.get_index(0, 3, 1)) = 0;
    p_obs_state(1, state_index.get_index(0, 3, 0)) = 0;
    p_obs_state(1, state_index.get_index(0, 3, 1)) = 1;

    p_obs_state(0, state_index.get_index(2, 3, 0)) = 1;
    p_obs_state(0, state_index.get_index(2, 3, 1)) = 0;
    p_obs_state(1, state_index.get_index(2, 3, 0)) = 0;
    p_obs_state(1, state_index.get_index(2, 3, 1)) = 1;

//    cout << p_obs_state;

    // 状态转移
    vector<Eigen::Matrix<double, 40, 40>> trans_vec;
    Eigen::Matrix<double, 40, 40> transition(40, 40);
    transition.setConstant(0);
    trans_vec.push_back(transition);
    trans_vec.push_back(transition);
    trans_vec.push_back(transition);
    trans_vec.push_back(transition);
    // 直接写吧，以后再封装
    Eigen::Matrix<int, 7, 6> map(7, 6);
    map.setConstant(0);
    map(2, 2) = 1;
    map(2, 3) = 1;
    map(4, 2) = 1;
    map(4, 3) = 1;
    map(2, 4) = 2;
    map(1, 3) = 3;
    map(3, 3) = 4;
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 6; j++) {
            if (i == 0 || j == 0 || i == 6 || j == 5) {
                map(i, j) = -1;
            }
        }
    }
    cout << map << endl;

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 4; j++) {
            // 检测当前栅格状态
            if (map(i + 1, j + 1) == 1 || map(i + 1, j + 1) == 2) {
                trans_vec[UP](state_index.get_index(i, j, 0), state_index.get_index(i, j, 0)) = 1;
                trans_vec[UP](state_index.get_index(i, j, 1), state_index.get_index(i, j, 1)) = 1;
                trans_vec[DOWN](state_index.get_index(i, j, 0), state_index.get_index(i, j, 0)) = 1;
                trans_vec[DOWN](state_index.get_index(i, j, 1), state_index.get_index(i, j, 1)) = 1;
                trans_vec[LEFT](state_index.get_index(i, j, 0), state_index.get_index(i, j, 0)) = 1;
                trans_vec[LEFT](state_index.get_index(i, j, 1), state_index.get_index(i, j, 1)) = 1;
                trans_vec[RIGHT](state_index.get_index(i, j, 0), state_index.get_index(i, j, 0)) = 1;
                trans_vec[RIGHT](state_index.get_index(i, j, 1), state_index.get_index(i, j, 1)) = 1;
                continue;
            }
            // UP
            switch (map(i + 1, j - 1 + 1)) {
                case 0:
                    trans_vec[UP](state_index.get_index(i, j, 0), state_index.get_index(i, j - 1, 0)) = 1;
                    trans_vec[UP](state_index.get_index(i, j, 1), state_index.get_index(i, j - 1, 1)) = 1;
                    break;
                case 1:
                    trans_vec[UP](state_index.get_index(i, j, 0), state_index.get_index(i, j, 0)) = 1;
                    trans_vec[UP](state_index.get_index(i, j, 1), state_index.get_index(i, j, 1)) = 1;
                    break;
                case 2:
                    trans_vec[UP](state_index.get_index(i, j, 0), state_index.get_index(i, j - 1, 0)) = 1;
                    trans_vec[UP](state_index.get_index(i, j, 1), state_index.get_index(i, j - 1, 1)) = 1;
                    break;
                case 3:
                    trans_vec[UP](state_index.get_index(i, j, 0), state_index.get_index(i, j, 0)) = 1;
                    trans_vec[UP](state_index.get_index(i, j, 1), state_index.get_index(i, j - 1, 1)) = 1;
                    break;
                case 4:
                    trans_vec[UP](state_index.get_index(i, j, 0), state_index.get_index(i, j - 1, 0)) = 1;
                    trans_vec[UP](state_index.get_index(i, j, 1), state_index.get_index(i, j, 1)) = 1;
                    break;
                case -1:
                    trans_vec[UP](state_index.get_index(i, j, 0), state_index.get_index(i, j, 0)) = 1;
                    trans_vec[UP](state_index.get_index(i, j, 1), state_index.get_index(i, j, 1)) = 1;
                    break;
            }
            // DOWN
            switch (map(i + 1, j + 1 + 1)) {
                case 0:
                    trans_vec[DOWN](state_index.get_index(i, j, 0), state_index.get_index(i, j + 1, 0)) = 1;
                    trans_vec[DOWN](state_index.get_index(i, j, 1), state_index.get_index(i, j + 1, 1)) = 1;
                    break;
                case 1:
                    trans_vec[DOWN](state_index.get_index(i, j, 0), state_index.get_index(i, j, 0)) = 1;
                    trans_vec[DOWN](state_index.get_index(i, j, 1), state_index.get_index(i, j, 1)) = 1;
                    break;
                case 2:
                    trans_vec[DOWN](state_index.get_index(i, j, 0), state_index.get_index(i, j + 1, 0)) = 1;
                    trans_vec[DOWN](state_index.get_index(i, j, 1), state_index.get_index(i, j + 1, 1)) = 1;
                    break;
                case 3:
                    trans_vec[DOWN](state_index.get_index(i, j, 0), state_index.get_index(i, j, 0)) = 1;
                    trans_vec[DOWN](state_index.get_index(i, j, 1), state_index.get_index(i, j + 1, 1)) = 1;
                    break;
                case 4:
                    trans_vec[DOWN](state_index.get_index(i, j, 0), state_index.get_index(i, j + 1, 0)) = 1;
                    trans_vec[DOWN](state_index.get_index(i, j, 1), state_index.get_index(i, j, 1)) = 1;
                    break;
                case -1:
                    trans_vec[DOWN](state_index.get_index(i, j, 0), state_index.get_index(i, j, 0)) = 1;
                    trans_vec[DOWN](state_index.get_index(i, j, 1), state_index.get_index(i, j, 1)) = 1;
                    break;
            }
            // LEFT
            switch (map(i - 1 + 1, j + 1)) {
                case 0:
                    trans_vec[LEFT](state_index.get_index(i, j, 0), state_index.get_index(i - 1, j, 0)) = 1;
                    trans_vec[LEFT](state_index.get_index(i, j, 1), state_index.get_index(i - 1, j, 1)) = 1;
                    break;
                case 1:
                    trans_vec[LEFT](state_index.get_index(i, j, 0), state_index.get_index(i, j, 0)) = 1;
                    trans_vec[LEFT](state_index.get_index(i, j, 1), state_index.get_index(i, j, 1)) = 1;
                    break;
                case 2:
                    trans_vec[LEFT](state_index.get_index(i, j, 0), state_index.get_index(i - 1, j, 0)) = 1;
                    trans_vec[LEFT](state_index.get_index(i, j, 1), state_index.get_index(i - 1, j, 1)) = 1;
                    break;
                case 3:
                    trans_vec[LEFT](state_index.get_index(i, j, 0), state_index.get_index(i, j, 0)) = 1;
                    trans_vec[LEFT](state_index.get_index(i, j, 1), state_index.get_index(i - 1, j, 1)) = 1;
                    break;
                case 4:
                    trans_vec[LEFT](state_index.get_index(i, j, 0), state_index.get_index(i - 1, j, 0)) = 1;
                    trans_vec[LEFT](state_index.get_index(i, j, 1), state_index.get_index(i, j, 1)) = 1;
                    break;
                case -1:
                    trans_vec[LEFT](state_index.get_index(i, j, 0), state_index.get_index(i, j, 0)) = 1;
                    trans_vec[LEFT](state_index.get_index(i, j, 1), state_index.get_index(i, j, 1)) = 1;
                    break;
            }
            // RIGHT
            switch (map(i + 1 + 1, j + 1)) {
                case 0:
                    trans_vec[RIGHT](state_index.get_index(i, j, 0), state_index.get_index(i + 1, j, 0)) = 1;
                    trans_vec[RIGHT](state_index.get_index(i, j, 1), state_index.get_index(i + 1, j, 1)) = 1;
                    break;
                case 1:
                    trans_vec[RIGHT](state_index.get_index(i, j, 0), state_index.get_index(i, j, 0)) = 1;
                    trans_vec[RIGHT](state_index.get_index(i, j, 1), state_index.get_index(i, j, 1)) = 1;
                    break;
                case 2:
                    trans_vec[RIGHT](state_index.get_index(i, j, 0), state_index.get_index(i + 1, j, 0)) = 1;
                    trans_vec[RIGHT](state_index.get_index(i, j, 1), state_index.get_index(i + 1, j, 1)) = 1;
                    break;
                case 3:
                    trans_vec[RIGHT](state_index.get_index(i, j, 0), state_index.get_index(i, j, 0)) = 1;
                    trans_vec[RIGHT](state_index.get_index(i, j, 1), state_index.get_index(i + 1, j, 1)) = 1;
                    break;
                case 4:
                    trans_vec[RIGHT](state_index.get_index(i, j, 0), state_index.get_index(i + 1, j, 0)) = 1;
                    trans_vec[RIGHT](state_index.get_index(i, j, 1), state_index.get_index(i, j, 1)) = 1;
                    break;
                case -1:
                    trans_vec[RIGHT](state_index.get_index(i, j, 0), state_index.get_index(i, j, 0)) = 1;
                    trans_vec[RIGHT](state_index.get_index(i, j, 1), state_index.get_index(i, j, 1)) = 1;
                    break;
            }
        }
    }

//    cout << trans_vec[UP];
    Eigen::MatrixXd gamma;
    // 第一列是动作
    gamma.conservativeResize(1,41);
    gamma.setConstant(0);

    // 计算Vh
    for (int horizon = 0; horizon < 3; horizon++) {
        Eigen::MatrixXd new_gamma;
        vector<vector<vector<vector<double>>>> tmp;
        tmp.resize(gamma.rows());
        for(int row = 0; row < gamma.rows(); row++){
            tmp[row].resize(4);
            for(int action = 0; action < 4; action++){
                tmp[row][action].resize(2);
            }
        }

        // 约束数量
        for (int row = 0; row < gamma.rows(); row++) {
            // 动作
            for (int action = 0; action < 4; action++) {
                // 观测
                for (int z = 0; z < 2; ++z) {
                    // 状态
                    for (int i = 0; i < 5; i++) {
                        for (int j = 0; j < 4; j++) {
                            for (int k = 0; k < 2; k++) {
                                double v = (gamma.row(row).rightCols(40).array() * p_obs_state.row(z).array()).matrix()
                                * trans_vec[action].transpose().col(state_index.get_index(i, j, k));
                                tmp[row][action][z].push_back(v);
                            }
                        }
                    }
                }
            }
        }

        // 动作
        for(int action = 0; action < 4; action++){
            // 排列组合
            for(int per_1 = 0; per_1 < gamma.rows(); per_1 ++){
                for(int per_2 = 0; per_2 < gamma.rows(); per_2 ++){
                    new_gamma.conservativeResize(new_gamma.rows()+1, 41);
                    new_gamma(new_gamma.rows()-1, 0) = action;
                    // 状态
                    for (int i = 0; i < 5; i++) {
                        for (int j = 0; j < 4; j++) {
                            for (int k = 0; k < 2; k++) {
                                double reward = -1;
                                if(i == 1 && j == 3) reward = 0;
                                double sum = tmp[per_1][action][0][state_index.get_index(i,j,k)]
                                        + tmp[per_2][action][1][state_index.get_index(i,j,k)];
                                new_gamma(new_gamma.rows()-1, state_index.get_index(i,j,k)+1) = reward + sum;
                            }
                        }
                    }
                }
            }
        }
        gamma = new_gamma;
    }
//    cout << gamma << endl;
    vector<vector<list<int>>> policy;

//    cout << "hello";
}
