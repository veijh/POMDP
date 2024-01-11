//
// Created by wjh on 24-1-11.
//
//
// Created by wjh on 24-1-2.
//

#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif

#ifndef EIGEN_VECTORIZE_SSE4_2
#define EIGEN_VECTORIZE_SSE4_2
#endif

#define CRASH_REWARD (-100)

#pragma GCC optimize(3)

#include "iostream"
#include <iomanip>
#include "vector"
#include "omp.h"
#include "Eigen/Eigen"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "unordered_map"
#include "POMDP.h"
#include "single_UAV_maze.h"

typedef struct Node {
    int id;
    Eigen::Vector2d pos;
    unordered_map<int, double> edge_list;
}Node;

using namespace std;

// dim: C(n,r) x r
vector<vector<int>> generateCombinations(int n, int r) {
    vector<vector<int>> output;
    if(r == 0) return output;
    vector<int> combination(r, 0);

    // 初始化组合的初始状态
    for (int i = 0; i < r; ++i) {
        combination[i] = i;
    }

    while (combination[0] <= n - r) {
        // 输出当前组合
//        for (int num : combination) {
//            cout << num << " ";
//        }
//        cout << endl;
        output.push_back(combination);

        // 生成下一个组合
        int i = r - 1;
        while (i >= 0 && combination[i] == n - r + i) {
            --i;
        }

        if (i >= 0) {
            ++combination[i];
            for (int j = i + 1; j < r; ++j) {
                combination[j] = combination[j - 1] + 1;
            }
        } else {
            break;  // 已生成所有组合
        }
    }
    return output;
}

// get the bit of value, index:[....76543210]
unsigned int get_bit(int value, int r){
    return (value >> r) & 1;
}

int main() {
#if _OPENMP
    cout << " support openmp " << endl;
#else
    cout << " not support openmp" << endl;
#endif
    omp_set_num_threads(6);
    int obs_dim = 3, state_dim = 84*64;
    int act_dim = 0;

    // init POMDP
    vector<Eigen::MatrixXf> tran_vec;
    Eigen::MatrixXf tran(state_dim, state_dim);
    tran.setConstant(0);
    for(int i = 0; i < act_dim; i++){
        tran_vec.push_back(tran);
    }
    Eigen::MatrixXf reward = Eigen::MatrixXf::Zero(state_dim, act_dim);
    Eigen::MatrixXf p_o_s = Eigen::MatrixXf::Zero(obs_dim, state_dim);
    for(int s = 0; s < state_dim; s++){
        for(int act = 0; act < act_dim; act++){
            // every node has 64 states
            // dst node is an absorbed state
            if(s/64 == 3 || s/64 == 4 || s/64 == 5){
                tran_vec[act](s, s) = 1;
                reward(s, act) = 0;
                continue;
            }
            // normal node
            if(act < adj_table[s/64].edge_list.size()){
                auto header = adj_table[s/64].edge_list.begin();
                for(int mv = 0; mv < act; mv++, header++);
                tran_vec[act](s, header->first) = 1;
                reward(s, act) = -header->second;
            }
            else{
                tran_vec[act](s, s) = 1;
                reward(s, act) = CRASH_REWARD;
            }
        }
        // p_o_s
        if (var.find(s / 64) != var.end()) {
            if (((s % 64) >> var[s / 64] & 1) == 0) {
                p_o_s(0, s) = 1;
                p_o_s(1, s) = 0;
                p_o_s(2, s) = 0;
            }
            else {
                p_o_s(0, s) = 0;
                p_o_s(1, s) = 1;
                p_o_s(2, s) = 0;
            }
        }
        else {
            p_o_s(0, s) = 0;
            p_o_s(1, s) = 0;
            p_o_s(2, s) = 1;
        }
    }

    POMDP PBVI(tran_vec, reward, p_o_s);

    // PBVI的核心
    int node_state_num = (int)pow(3,6);
    const int point_num = node_state_num*state_dim/64;
    Eigen::MatrixXf possible_state(64, node_state_num);
    possible_state.setZero();
    // C(6,r)
    int count = 0;
    // 生成组合数索引，重点关注对象
    for(int r = 0; r <= 6; r++){
        if(r == 0){
            possible_state.col(count) = (float)pow(0.5, 6) * Eigen::MatrixXf::Ones(64,1);
            count++;
            continue;
        }
        auto C_n_r = generateCombinations(6, r);
        for(auto item:C_n_r){
            // 生成匹配的掩码，重点关注对象的所有可能的情况
            for(int mask = 0; mask < (int)pow(2,r); mask++){
                // 标记匹配掩码的索引
                for(unsigned int index = 0; index < 64; index++){
                    // 检查index能否匹配
                    bool is_matched = true;
                    for(int bit_index = 0; bit_index < r; bit_index++){
                        unsigned int bit = 0;
                        bit = get_bit(mask, bit_index);
                        if(bit != get_bit(index, item[bit_index])){
                            is_matched = false;
                            break;
                        }
                    }
                    // index如果能匹配，对应概率为1/2^(6-r)
                    if(is_matched) {
                        possible_state(index, count) = (float)pow(0.5, 6-r);
                    }
                }
                count++;
            }
        }
    }

    // 信念点的集合 N x point_num
//    Eigen::MatrixXf belief_point(state_dim, point_num);
    Eigen::SparseMatrix<float> belief_point(1+state_dim, point_num);
    belief_point.setZero();
    for(int i = 0; i < state_dim/64; i++){
//        belief_point.block(64*i,node_state_num*i,possible_state.rows(),possible_state.cols()) = possible_state;
        Eigen::MatrixXf belief_point_col = Eigen::MatrixXf::Zero(1+state_dim, node_state_num);
        belief_point_col.middleRows(1+64*i, 64) = possible_state;
        belief_point.middleCols(node_state_num*i, node_state_num) = belief_point_col.sparseView();
    }

    PBVI.PBVI(belief_point, 100);
}
