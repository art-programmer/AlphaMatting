#ifndef COST_FUNCTOR_H__
#define COST_FUNCTOR_H__

#include <memory>

class CostFunctor {
public:
  typedef std::shared_ptr<CostFunctor> Ptr;

  virtual double operator()(const int node_index, const long label) const = 0;
  virtual double operator()(const int node_index_1, const int node_index_2,
                            const long label_1, const long label_2) const = 0;
  virtual double at(const int node_index, const long label) const = 0;
  virtual double at(const int node_index_1, const int node_index_2,
                    const long label_1, const long label_2) const = 0;

  virtual double getLabelCost() const { return 0; };
  virtual double getLabelIndicatorConflictCost() const { return 0; };
};

#endif
