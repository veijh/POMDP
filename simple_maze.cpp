//
// Created by wjh on 24-1-2.
//

#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif

#ifndef EIGEN_VECTORIZE_SSE4_2
#define EIGEN_VECTORIZE_SSE4_2
#endif

#define CRASH_REWARD (-999)

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
#include "maze_map.h"

using namespace std;

// return x^y
int my_pow(const int& x, const int& y){
    int ans = 1;
    for(int i = 0; i < y; i++){
        ans *= x;
    }
    return ans;
}

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
    cout << "support openmp " << endl;
#else
    cout << "not support openmp" << endl;
#endif
    omp_set_num_threads(6);

    // read topo from file
    FILE* node_file = fopen("../node.csv", "r");
    FILE* edge_file = fopen("../edge.csv", "r");

    if(node_file == nullptr) {
        cout << "[ERROR] fail to open node file!" << endl;
        return 0;
    }

    if(edge_file == nullptr) {
        cout << "[ERROR] fail to open edge file!" << endl;
        return 0;
    }

    Map map;
    double x = 0.0, y = 0.0;
    int id = 0;
    while (~fscanf(node_file, "%lf,%lf,%d", &x, &y, &id)) {
        map.adj_table[id].pos << x,y;
    }

    int id1 = 0, id2 = 0;
    while (~fscanf(edge_file, "%d,%d", &id1, &id2)) {
        map.add_edge(id1, id2);
    }
    cout << "[INFO] the number of nodes is " << map.get_node_num() << endl;

    fclose(node_file);
    fclose(edge_file);

    /*
    for(auto node:map.adj_table){
        cout << "node " << node.first << " >> ";
        for(auto item:node.second.edge_list){
            cout << item.first << ": " << item.second << "; ";
        }
        cout << endl;
    }
    */

    // unknown door
    vector<vector<int>> unk_part{{13,12,14},
                                     {16,17,15,18},{25,26,27,28}};
    const int doors_num = unk_part.size();
    const int all_condition_num = my_pow(2, doors_num);
    cout << "[INFO] the number of unknown doors is " << doors_num << endl;

    unordered_map<int, int> var;
    vector<int> unk_node;
    for(int i = 0; i < doors_num; i++){
        for(auto item:unk_part[i]){
            var[item] = i;
            unk_node.push_back(item);
        }
    }

    vector<int> src = {0, 1, 2}, dst = {3, 4, 5}, key_node;
    key_node.insert(key_node.end(), src.begin(), src.end());
    key_node.insert(key_node.end(), dst.begin(), dst.end());
    key_node.insert(key_node.end(), unk_node.begin(), unk_node.end());

    Map compact_map;
    for(int node:key_node){
        compact_map.adj_table[node].pos = map.adj_table[node].pos;
    }
    cout << "[INFO] the number of key nodes is " << compact_map.get_node_num() << endl;

    vector<int> path;
    for(int begin:key_node){
        for (int end:key_node){
            if(end == begin) continue;
            double dis = map.dijkstra(begin, {end}, path);
            if(path.size() != 0){
                compact_map.add_edge(begin, end, dis);
            }
        }
    }

    /*
    for(auto node:compact_map.adj_table){
        cout << "node " << node.first << " >> ";
        for(auto item:node.second.edge_list){
            cout << item.first << ": " << item.second << "; ";
        }
        cout << endl;
    }
     */

    cout << "[LOG] determine the act dim" << endl;
    Map fc_compact_map(compact_map);
    for(auto item:unk_part){
        int size = item.size();
        fc_compact_map.add_edge(item[0], item[1]);
        fc_compact_map.add_edge(item[size-2], item[size-1]);
    }


    for(auto node:fc_compact_map.adj_table){
        cout << "node " << node.first << ":";
        for(auto item:node.second.edge_list){
            cout << "act[" << item.first << "]:" << item.second << "; ";
        }
        cout << endl;
    }


    int act_dim = 0;
    for(auto item:fc_compact_map.adj_table){
        if(item.second.edge_list.size() > act_dim) {
            act_dim = item.second.edge_list.size();
        }
    }
    cout << "[INFO] act_dim is " << act_dim << endl;

    // all kinds of possible structure
    vector<Map> all_compact_map(all_condition_num, compact_map);

    for(int i = 0; i < all_compact_map.size(); i++){
        for(int bit = 0; bit < doors_num; bit++){
            int id_c1, id_c2;
            if( ((i >> bit) & 1) == 0){
                id_c1 = unk_part[bit][0];
                id_c2 = unk_part[bit][1];
            }
            else{
                id_c1 = unk_part[bit][unk_part[bit].size()-2];
                id_c2 = unk_part[bit][unk_part[bit].size()-1];
            }

            double dis = (compact_map.adj_table[id_c1].pos - compact_map.adj_table[id_c2].pos).norm();
            all_compact_map[i].adj_table[id_c1].edge_list[id_c2] = dis;
            all_compact_map[i].adj_table[id_c2].edge_list[id_c1] = dis;
        }
    }

    int obs_dim = 3, state_dim = compact_map.get_node_num()*all_condition_num+1;
    cout << "[INFO] obs_dim = " << obs_dim << ", state_dim = " << state_dim  << endl;
    vector<int> state_map(compact_map.get_node_num());
    unordered_map<int,int> inv_state_map;
    for(int i = 0; i < state_map.size(); i++){
        auto header = compact_map.adj_table.begin();
        for(int j = 0; j < i; j++, header++);
        state_map[i] = header->first;
        inv_state_map[header->first] = i;
    }

//    for(auto item:state_map){
//        cout << item << ", ";
//    }

//    for(auto item:inv_state_map){
//        cout << item.first << ", " << item.second << endl;
//    }

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
            // the last state is a crashed state
            if(s == state_dim - 1) {
                tran_vec[act](s, s) = 1;
                reward(s, act) = 0;
                continue;
            }
            // every node else has all_condition_num states
            int real_node = state_map[s/all_condition_num];
            // dst node is an absorbed state
            if(real_node == 3 || real_node == 4 || real_node == 5){
                tran_vec[act](s, s) = 1;
                reward(s, act) = 0;
                continue;
            }
            // normal node
            int dst_id = 0;
            if(act < fc_compact_map.adj_table[real_node].edge_list.size()){
                auto header = fc_compact_map.adj_table[real_node].edge_list.begin();
                for(int mv = 0; mv < act; mv++, header++);
                dst_id = header->first;
                // search for dst_id according to s%all_condition_num map
                auto it = all_compact_map[s%all_condition_num].adj_table[real_node].edge_list;
                if(it.find(dst_id) != it.end()){
                    tran_vec[act](s, all_condition_num*inv_state_map[dst_id] + s%all_condition_num) = 1;
                    reward(s, act) = -header->second;
                }
                else
                {
                    tran_vec[act](s, state_dim-1) = 1;
                    reward(s, act) = CRASH_REWARD;
                }
            }
            else{
                tran_vec[act](s, state_dim-1) = 1;
                reward(s, act) = CRASH_REWARD;
            }
        }
        // p_o_s
        // s is unk_node
        if (s != state_dim-1 && var.find(state_map[s/all_condition_num]) != var.end()) {
            if ( ( ( (s % all_condition_num) >> var[state_map[s/all_condition_num]]) & 1) == 0) {
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
        // normal node, i.e. src, dst
        else {
            p_o_s(0, s) = 0;
            p_o_s(1, s) = 0;
            p_o_s(2, s) = 1;
        }
    }

    // init
    cout << "[LOG] start to init POMDP" << endl;
    POMDP PBVI(tran_vec, reward, p_o_s);

    // PBVI的核心
    int node_state_num = my_pow(3,doors_num);
    const int point_num = node_state_num*(state_dim-1)/all_condition_num + 1;
    Eigen::MatrixXf possible_state(all_condition_num, node_state_num);
    possible_state.setZero();
    // C(doors_num,r)
    int count = 0;
    // 生成组合数索引，重点关注对象
    for(int r = 0; r <= doors_num; r++){
        if(r == 0){
            possible_state.col(count) = (float)pow(0.5, doors_num) * Eigen::MatrixXf::Ones(all_condition_num,1);
            count++;
            continue;
        }
        auto C_n_r = generateCombinations(doors_num, r);
        for(auto item:C_n_r){
            // 生成匹配的掩码，重点关注对象的所有可能的情况
            for(int mask = 0; mask < my_pow(2,r); mask++){
                // 标记匹配掩码的索引
                for(unsigned int index = 0; index < all_condition_num; index++){
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
                    // index如果能匹配，对应概率为1/2^(doors_num-r)
                    if(is_matched) {
                        possible_state(index, count) = (float)pow(0.5, doors_num-r);
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
    for(int i = 0; i < (state_dim-1)/all_condition_num; i++){
//        belief_point.block(all_condition_num*i,node_state_num*i,possible_state.rows(),possible_state.cols()) = possible_state;
        Eigen::MatrixXf belief_point_col = Eigen::MatrixXf::Zero(1+state_dim, node_state_num);
        belief_point_col.middleRows(1+all_condition_num*i, all_condition_num) = possible_state;
        belief_point.middleCols(node_state_num*i, node_state_num) = belief_point_col.sparseView();
    }
    belief_point.insert(state_dim, point_num-1) = 1;
    belief_point.makeCompressed();

    PBVI.PBVI(belief_point, 200);

}
