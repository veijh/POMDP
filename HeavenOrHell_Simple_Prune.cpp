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

//    cout << trans_vec[UP];
    Eigen::MatrixXd gamma;
    // 第一列是动作
    gamma.conservativeResize(1, 1 + total_state);
    gamma.setConstant(0);

    // 计算Vh
    for (int horizon = 0; horizon < 10; horizon++) {
        cout << horizon << endl;
        Eigen::MatrixXd new_gamma;
        vector<vector<vector<vector<double>>>> tmp;
        tmp.resize(gamma.rows());
        for (int row = 0; row < gamma.rows(); row++) {
            tmp[row].resize(4);
            for (int action = 0; action < 4; action++) {
                tmp[row][action].resize(2);
            }
        }

        // 这一段可以并行计算
        // 约束数量
        for (int row = 0; row < gamma.rows(); row++) {
            // 动作
            for (int action = 0; action < 4; action++) {
                // 观测
                for (int z = 0; z < 2; ++z) {
                    // 状态
                    for (int s = 0; s < total_state; s++) {
                        double v = (gamma.row(row).rightCols(total_state).array() * p_obs_state.row(z).array()).matrix()
                                   * trans_vec[action].transpose().col(s);
                        tmp[row][action][z].push_back(v);
                    }
                }
            }
        }

        new_gamma.conservativeResize(4 * gamma.rows() * gamma.rows(), 1 + total_state);
        // 动作
        for (int action = 0; action < 4; action++) {
            // 排列组合
            for (int per_1 = 0; per_1 < gamma.rows(); per_1++) {
                for (int per_2 = 0; per_2 < gamma.rows(); per_2++) {
                    new_gamma(per_2 + gamma.rows() * per_1 + gamma.rows() * gamma.rows() * action, 0) = action;
                    // 状态
                    for (int s = 0; s < total_state; ++s) {
                        double reward = -1;
                        if (s == 4 && action == UP) reward = 100;
                        if (s == 5 && action == UP) reward = -100;
                        if (s == 8 && action == UP) reward = -100;
                        if (s == 9 && action == UP) reward = 100;
                        if (s == 0 || s == 1 || s == 2 || s == 3) reward = 0;
                        double sum = tmp[per_1][action][0][s]
                                     + tmp[per_2][action][1][s];
                        new_gamma(per_2 + gamma.rows() * per_1 + gamma.rows() * gamma.rows() * action, s + 1) = reward + sum;
                    }
                }
            }
        }

        // 随机采样对非积极约束剪枝
        const int sample_num = 20000;
        int active_num = 0;
        Eigen::MatrixXd sample_point(1+total_state, sample_num);
        sample_point.setConstant(0);
        sample_point.bottomRows(total_state) = Eigen::MatrixXd::Random(total_state, sample_num)+Eigen::MatrixXd::Ones(total_state,sample_num);
        sample_point = (sample_point.array()/(sample_point.colwise().sum().array().replicate<1+total_state, 1>())).matrix();

        Eigen::MatrixXd ans = new_gamma * sample_point;
        Eigen::MatrixXd gamma_prune;
        vector<int> active_con_index(new_gamma.rows(),0);
        for(int col = 0; col < sample_num; col ++){
            int index = 0;
            ans.col(col).maxCoeff(&index);
            if(active_con_index[index] == 0){
                active_con_index[index] = 1;
                active_num++;
            }
        }

        gamma_prune.conservativeResize(active_num, 1+total_state);
        int gamma_prune_index = 0;
        for(int i = 0; i < active_con_index.size(); i++){
            if(active_con_index[i]){
                gamma_prune.row(gamma_prune_index) = new_gamma.row(i);
                gamma_prune_index++;
            }
        }

        gamma = gamma_prune;
    }
    cout << gamma << endl;

    vector<set<int>> policy;
    for(int s = 0; s < total_state/2; s++){
        Eigen::VectorXd belief(1+total_state);
        belief.setConstant(0);
        belief(2*s + 1) = 0.5;
        belief(2*s+1 + 1) = 0.5;
        Eigen::Index index;
        auto result = gamma*belief;
        double max_v = result.maxCoeff();
        set<int> tmp;
        for(int i = 0; i < gamma.rows(); i++){
            if(max_v == result(i)){
                if (tmp.find(gamma(i,0)) == tmp.end()) {
                    tmp.insert(gamma(i,0));
                }
            }
        }
        policy.push_back(tmp);
    }

    for(auto i:policy){
        for(auto j:i){
            switch (j) {
                case UP:
                    cout << "^";
                    break;
                case LEFT:
                    cout << "<";
                    break;
                case RIGHT:
                    cout << ">";
                    break;
                case DOWN:
                    cout << "v";
                    break;
            }
        }
        cout << " ";

    }

//    cout << "hello";
}
