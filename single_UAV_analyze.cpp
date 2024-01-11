//
// Created by wjh on 24-1-10.
//

#include "single_UAV_analyze.h"

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
#include "single_UAV_analyze.h"
#include "cmath"

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
    vector<vector<int>> unk_part{{6,7,8},{9,10,11},{13,12,14},{33,35,34},
                                 {16,17,15,18},{25,26,27,28}};
    unordered_map<int, int> var;
    for(int i = 0; i < unk_part.size(); i++){
        for(auto item:unk_part[i]){
            var[item] = i;
        }
    }

    // all kinds of possible structure
    vector<vector<Node>> all_adj_table(64, adj_table);

    for(int i = 0; i < all_adj_table.size(); i++){
        for(int bit = 0; bit < 6; bit++){
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
    for(auto node:adj_table){
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

    Eigen::MatrixXf alpha_vector;
    MATSL::read_binary("../output.bin", alpha_vector);

    Eigen::VectorXf _belief_state(state_dim);
    Eigen::VectorXf node_belief_state = Eigen::VectorXf::Zero(64);
    node_belief_state(0) = 1;

    for(int node = 0; node < 84; node++){
        _belief_state.setConstant(0);
        _belief_state.middleRows(64*node, 64) = node_belief_state;

        Eigen::VectorXf adv_belief_state(1 + state_dim);
        adv_belief_state.block(1, 0, state_dim, 1) = _belief_state;
//        cout << adv_belief_state;
//        cout << "start to calculate result" << endl;
        Eigen::VectorXf result = alpha_vector * adv_belief_state;
//        cout << "start to search max_v" << endl;
        double max_v = result.maxCoeff();
//        cout << "start to match the best action" << endl;
        vector<int> best_actions;
        for(int i = 0; i < alpha_vector.rows(); i++){
            if(abs(max_v - result(i)) < 0.1) {
                if (find(best_actions.begin(), best_actions.end(),(int)alpha_vector(i,0)) == best_actions.end()) {
                    best_actions.push_back((int)alpha_vector(i,0));
                }
            }
        }

        cout << node << ": ";
        for(auto act:best_actions){
            cout << act << ",";
            if(act < adj_table[node].edge_list.size()){
                auto header = adj_table[node].edge_list.begin();
                for(int mv = 0; mv < act; mv++, header++);
                cout << header->first << "; ";
            }
        }
        cout << endl;
    }

}
