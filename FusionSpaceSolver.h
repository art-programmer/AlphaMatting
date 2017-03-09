#ifndef FUSION_SPACE_SOLVER_H__
#define FUSION_SPACE_SOLVER_H__

#include <FusionSolver.h>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

#include "AlphaMattingProposalGenerator.h"
#include "CostFunctor.h"
#include "ProposalGenerator.h"

class FusionSpaceSolver : public ParallelFusion::FusionSolver<
                              AlphaMattingProposalGenerator::LABELSPACE> {
public:
  typedef AlphaMattingProposalGenerator::LABELSPACE LABELSPACE;

  FusionSpaceSolver(const int NUM_NODES,
                    const std::shared_ptr<const std::vector<std::vector<int>>>,
                    CostFunctor::Ptr cost_functor,
                    const std::chrono::nanoseconds &timeout,
                    const int NUM_ITERATIONS = 1000,
                    const bool CONSIDER_LABEL_COST = false);

  typedef std::shared_ptr<FusionSpaceSolver> Ptr;
  typedef std::shared_ptr<const FusionSpaceSolver> ConstPtr;
  template <typename... Targs> static inline Ptr Create(Targs &... args) {
    return std::make_shared<FusionSpaceSolver>(std::forward<Targs>(args)...);
  };
  template <typename... Targs> static inline Ptr Create(Targs &&... args) {
    return std::make_shared<FusionSpaceSolver>(std::forward<Targs>(args)...);
  };

  //  void setNeighbors();
  // void setNeighbors(const int width, const int height, const int
  // neighbor_system = 8);

  virtual void
  solve(const LABELSPACE &proposals,
        const ParallelFusion::SolutionType<LABELSPACE> &current_solution,
        ParallelFusion::SolutionType<LABELSPACE> &solution);
  virtual double evaluateEnergy(const LABELSPACE &solution) const {
    return 1e10;
  }

private:
  const int NUM_NODES_;
  const int NUM_ITERATIONS_;
  const bool CONSIDER_LABEL_COST_;

  CostFunctor::Ptr cost_functor_;
  const std::shared_ptr<const std::vector<std::vector<int>>> node_neighbors_;
  LABELSPACE fuse(const LABELSPACE &proposal_labels,
                  std::vector<double> &energy_info);

  std::mutex mtx;
  std::condition_variable cv;
  std::atomic<int> num_threads;
  const std::chrono::nanoseconds timeout;
};

#endif
