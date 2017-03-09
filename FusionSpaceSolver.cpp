#include "FusionSpaceSolver.h"

#include <iostream>
#include <limits>
#include <map>
#include <memory>

#include "TRW_S/MRFEnergy.h"

using namespace std;

FusionSpaceSolver::FusionSpaceSolver(
    const int NUM_NODES,
    const std::shared_ptr<const std::vector<std::vector<int>>> node_neighbors,
    CostFunctor::Ptr cost_functor, const std::chrono::nanoseconds &timeout,
    const int NUM_ITERATIONS, const bool CONSIDER_LABEL_COST)
    : NUM_NODES_(NUM_NODES), node_neighbors_(node_neighbors),
      cost_functor_(cost_functor), NUM_ITERATIONS_(NUM_ITERATIONS),
      CONSIDER_LABEL_COST_(CONSIDER_LABEL_COST), timeout(timeout),
      num_threads(4) {}

FusionSpaceSolver::LABELSPACE
FusionSpaceSolver::fuse(const LABELSPACE &node_labels,
                        std::vector<double> &energy_info) {
#define THROTTLE 1

#if THROTTLE
  {
    std::unique_lock<std::mutex> lk(mtx);
    if (num_threads == 0)
      cv.wait(lk, [&]() -> bool { return num_threads != 0; });

    num_threads -= 1;
  }
#endif

  vector<long> fused_labels;

  {
    LOG(INFO) << "fusing" << endl;

    unique_ptr<MRFEnergy<TypeGeneral>> energy(
        new MRFEnergy<TypeGeneral>(TypeGeneral::GlobalSize()));
    map<int, int> label_indicator_index_map;
    if (CONSIDER_LABEL_COST_) {
      int label_indicator_index = NUM_NODES_;
      for (auto node_it = node_labels.begin(); node_it != node_labels.end();
           node_it++)
        for (auto label_it = node_it->begin(); label_it != node_it->end();
             label_it++)
          if (label_indicator_index_map.count(*label_it) == 0)
            label_indicator_index_map[*label_it] = label_indicator_index++;
    }
    int NUM_LABEL_INDICATORS = label_indicator_index_map.size();
    vector<MRFEnergy<TypeGeneral>::NodeId> nodes(NUM_NODES_ +
                                                 NUM_LABEL_INDICATORS);

    // add unary cost
    for (int node_index = 0; node_index < NUM_NODES_; node_index++) {
      vector<long> labels = node_labels[node_index];
      const int NUM_LABELS = labels.size();
      if (NUM_LABELS == 0) {
        cout << "empty proposal error: " << node_index << endl;
        exit(1);
      }
      vector<double> unary_cost(NUM_LABELS);
      for (int label_index = 0; label_index < NUM_LABELS; label_index++)
        unary_cost[label_index] =
            cost_functor_->at(node_index, labels[label_index]);
      nodes[node_index] =
          energy->AddNode(TypeGeneral::LocalSize(NUM_LABELS),
                          TypeGeneral::NodeData(&unary_cost[0]));
    }

    // add label indicator cost
    if (CONSIDER_LABEL_COST_ == true) {
      for (int label_indicator_index = 0;
           label_indicator_index < NUM_LABEL_INDICATORS;
           label_indicator_index++) {
        vector<double> label_cost(2, 0);
        label_cost[1] = cost_functor_->getLabelCost();
        nodes[label_indicator_index + NUM_NODES_] = energy->AddNode(
            TypeGeneral::LocalSize(2), TypeGeneral::NodeData(&label_cost[0]));
      }
    }

    // add pairwise cost
    for (int node_index = 0; node_index < NUM_NODES_; node_index++) {
      vector<long> labels = node_labels[node_index];
      vector<int> neighbors = node_neighbors_->at(node_index);
      for (vector<int>::const_iterator neighbor_it = neighbors.begin();
           neighbor_it != neighbors.end(); neighbor_it++) {
        vector<long> neighbor_labels = node_labels[*neighbor_it];
        vector<double> pairwise_cost(labels.size() * neighbor_labels.size(), 0);
        for (int label_index = 0; label_index < labels.size(); label_index++)
          for (int neighbor_label_index = 0;
               neighbor_label_index < neighbor_labels.size();
               neighbor_label_index++)
            pairwise_cost[label_index + neighbor_label_index * labels.size()] =
                cost_functor_->at(node_index, *neighbor_it, labels[label_index],
                                  neighbor_labels[neighbor_label_index]);
        bool has_non_zero_cost = false;
        for (int i = 0; i < pairwise_cost.size(); i++)
          if (pairwise_cost[i] > 0)
            has_non_zero_cost = true;
        if (has_non_zero_cost == true) {
          // cout << node_index << *neighbor_it << endl;
          energy->AddEdge(
              nodes[node_index], nodes[*neighbor_it],
              TypeGeneral::EdgeData(TypeGeneral::GENERAL, &pairwise_cost[0]));
        }
      }
    }
    // exit(1);

    // add label indicator constraints
    if (CONSIDER_LABEL_COST_) {
      for (int node_index = 0; node_index < NUM_NODES_; node_index++) {
        vector<long> labels = node_labels[node_index];
        for (int label_index = 0; label_index < labels.size(); label_index++) {
          long label = labels[label_index];
          int label_indicator_index = label_indicator_index_map[label];
          vector<double> label_indicator_conflict_cost(labels.size() * 2, 0);
          label_indicator_conflict_cost[label_index] =
              cost_functor_->getLabelIndicatorConflictCost();

          energy->AddEdge(
              nodes[node_index], nodes[label_indicator_index + NUM_NODES_],
              TypeGeneral::EdgeData(TypeGeneral::GENERAL,
                                    &label_indicator_conflict_cost[0]));
        }
      }
    }

    MRFEnergy<TypeGeneral>::Options options;
    options.m_iterMax = NUM_ITERATIONS_;
    options.m_printIter = NUM_ITERATIONS_ / 5;
    options.m_printMinIter = 0;
    options.m_eps = 0.1;
    options.timeout = timeout;

    double lower_bound, solution_energy;
    energy->Minimize_TRW_S(options, lower_bound, solution_energy);

    fused_labels.resize(NUM_NODES_);
    for (int node_index = 0; node_index < NUM_NODES_; node_index++) {
      long label = energy->GetSolution(nodes[node_index]);
      fused_labels[node_index] = node_labels[node_index][label];
    }
    energy_info.assign(2, 0);
    energy_info[0] = solution_energy;
    energy_info[1] = lower_bound;
  }
#if THROTTLE
  num_threads += 1;
  cv.notify_all();
#endif
#undef THROTTLE
  return LABELSPACE(fused_labels);
}

void FusionSpaceSolver::solve(
    const LABELSPACE &proposals,
    const ParallelFusion::SolutionType<LABELSPACE> &current_solution,
    ParallelFusion::SolutionType<LABELSPACE> &solution) {

  vector<double> energy_info;
  solution.second = fuse(proposals, energy_info);
  solution.first = energy_info[0];
}
