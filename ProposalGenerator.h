#ifndef PROPOSAL_GENERATOR_H__
#define PROPOSAL_GENERATOR_H__

#include <vector>


class ProposalGenerator
{
 public:
  virtual void setCurrentSolution(const std::vector<long> &current_solution) = 0;
  virtual std::vector<std::vector<long> > getProposal() const = 0;
  
 protected:
  std::vector<long> current_solution_;
};

#endif
