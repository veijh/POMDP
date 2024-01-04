//
// Created by wjh on 24-1-2.
//

#include "iostream"
#include <iomanip>
#include "vector"
#include "omp.h"
#include "Eigen/Eigen"
#include "Eigen/Dense"
#include "unordered_map"
#include "POMDP.h"
#include "single_UAV_maze.h"

#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif

#ifndef EIGEN_VECTORIZE_SSE4_2
#define EIGEN_VECTORIZE_SSE4_2
#endif

#define CRASH_REWARD (-100)

//#pragma GCC optimize(3)

typedef struct Node {
    int id;
    Eigen::Vector2d pos;
    unordered_map<int, double> edge_list;
}Node;

using namespace std;

int main() {
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
}
