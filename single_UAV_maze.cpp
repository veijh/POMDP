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
    int obs_dim = 3, state_dim = 84*512;
    int act_dim = 0;

    // read topo from file
    FILE* node_file = fopen("../node.csv", "r");
    FILE* edge_file = fopen("../edge.csv", "r");

    if(node_file == nullptr) {
        cout << "fail to open node file!" << endl;
        return 0;
    }

    if(edge_file == nullptr) {
        cout << "fail to open edge file!" << endl;
        return 0;
    }

    vector<Node> adj_table(84);
    double x = 0.0, y = 0.0;
    int id = 0;
    while (~fscanf(node_file, "%lf,%lf,%d", &x, &y, &id)) {
        adj_table[id].id = id;
        adj_table[id].pos << x,y;
    }

    int id1 = 0, id2 = 0;
    while (~fscanf(edge_file, "%d,%d", &id1, &id2)) {
        if(adj_table[id1].edge_list.find(id2) == adj_table[id1].edge_list.end()){
            adj_table[id1].edge_list[id2] = (adj_table[id1].pos - adj_table[id2].pos).norm();
        }
        else{
            cout << "duplicate edge: (" << id1 << ", " << id2 << ")" << endl;
        }
        if(adj_table[id2].edge_list.find(id1) == adj_table[id2].edge_list.end()){
            adj_table[id2].edge_list[id1] = (adj_table[id1].pos - adj_table[id2].pos).norm();
        }
        else{
            cout << "duplicate edge: (" << id1 << ", " << id2 << ")" << endl;
        }
    }

    fclose(node_file);
    fclose(edge_file);

    // unknown door
    vector<vector<int>> unk_part{{6,7,8},{23,22,24},{9,10,11},{13,12,14},{21,19,20},{33,35,34},
                                     {16,17,15,18},{25,26,27,28},{29,31,30,32}};
    unordered_map<int, int> var;
    for(int i = 0; i < unk_part.size(); i++){
        for(auto item:unk_part[i]){
            var[item] = i;
        }
    }

    // all kinds of possible structure
    vector<vector<Node>> all_adj_table(512, adj_table);

    for(int i = 0; i < all_adj_table.size(); i++){
        for(int bit = 0; bit < 9; bit++){
            int id_c1, id_c2;
            if( ((i >> bit) & 1) == 0){
                id_c1 = unk_part[bit][0];
                id_c2 = unk_part[bit][1];
            }
            else{
                id_c1 = unk_part[bit][unk_part[bit].size()-2];
                id_c2 = unk_part[bit][unk_part[bit].size()-1];
            }

            double dis = (adj_table[id_c1].pos - adj_table[id_c2].pos).norm();

            if(all_adj_table[i][id_c1].edge_list.find(id_c2) == all_adj_table[i][id_c1].edge_list.end()){
                all_adj_table[i][id_c1].edge_list[id_c2] = dis;
            }
            else{
                cout << "duplicate edge: (" << id_c1 << ", " << id_c2 << ")" << endl;
            }
            if(all_adj_table[i][id_c2].edge_list.find(id_c1) == all_adj_table[i][id_c2].edge_list.end()){
                all_adj_table[i][id_c2].edge_list[id_c1] = dis;
            }
            else{
                cout << "duplicate edge: (" << id_c1 << ", " << id_c2 << ")" << endl;
            }

        }
    }

    /*
    for(auto node:table){
        cout << "node " << node.id << " >> ";
        if(node.edge_list.size() > act_dim){
            act_dim = node.edge_list.size();
        }
        for(auto item:node.edge_list){
            cout << item.first << ": " << item.second << "; ";
        }
        cout << endl;
    }
     */

    // determine the act_dim
    for(auto table:all_adj_table){
        for(auto node:table){
            if(node.edge_list.size() > act_dim){
                act_dim = node.edge_list.size();
            }
        }
    }

    cout << "act_dim: " << act_dim << endl;

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
            // every node has 512 states
            // dst node is an absorbed state
            if(s/512 == 3 || s/512 == 4 || s/512 == 5){
                tran_vec[act](s, s) = 1;
                reward(s, act) = 0;
                continue;
            }
            // normal node
            if(act < adj_table[s/512].edge_list.size()){
                auto header = adj_table[s/512].edge_list.begin();
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
        if (var.find(s / 512) != var.end()) {
            if (((s % 512) >> var[s / 512] & 1) == 0) {
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
    int node_state_num = (int)pow(3,9);
    const int point_num = node_state_num*state_dim/512;
    Eigen::MatrixXf possible_state(512, node_state_num);
    possible_state.setZero();
    // C(9,r)
    int count = 0;
    // 生成组合数索引，重点关注对象
    for(int r = 0; r <= 9; r++){
        if(r == 0){
            possible_state.col(count) = (float)pow(0.5, 9) * Eigen::MatrixXf::Ones(512,1);
            count++;
            continue;
        }
        auto C_n_r = generateCombinations(9, r);
        for(auto item:C_n_r){
            // 生成匹配的掩码，重点关注对象的所有可能的情况
            for(int mask = 0; mask < (int)pow(2,r); mask++){
                // 标记匹配掩码的索引
                for(unsigned int index = 0; index < 512; index++){
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
                    // index如果能匹配，对应概率为1/2^(9-r)
                    if(is_matched) {
                        possible_state(index, count) = (float)pow(0.5, 9-r);
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
    for(int i = 0; i < state_dim/512; i++){
//        belief_point.block(512*i,node_state_num*i,possible_state.rows(),possible_state.cols()) = possible_state;
        Eigen::MatrixXf belief_point_col = Eigen::MatrixXf::Zero(1+state_dim, node_state_num);
        belief_point_col.middleRows(1+512*i, 512) = possible_state;
        belief_point.middleCols(node_state_num*i, node_state_num) = belief_point_col.sparseView();
    }

    PBVI.PBVI(belief_point, 100);
}
