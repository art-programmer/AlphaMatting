#pragma once

#include <LabelSpace.h>
#include <glog/logging.h>

template <typename LabelType>
class AlphaMattingLabelSpace : public ParallelFusion::LabelSpace<LabelType> {
public:
  AlphaMattingLabelSpace() : ParallelFusion::LabelSpace<LabelType>(){};
  AlphaMattingLabelSpace(const int NUM_NODES)
      : ParallelFusion::LabelSpace<LabelType>(NUM_NODES){};

  AlphaMattingLabelSpace(const std::vector<LabelType> &single_solution)
      : ParallelFusion::LabelSpace<LabelType>(single_solution){};

  AlphaMattingLabelSpace(const std::vector<std::vector<LabelType>> &label_space)
      : ParallelFusion::LabelSpace<LabelType>(label_space){};

  std::vector<LabelType> &operator[](size_t idx) {
    CHECK_LT(idx, this->label_space_.size());
    return this->label_space_[idx];
  }

  const std::vector<LabelType> &operator[](size_t idx) const {
    CHECK_LT(idx, this->label_space_.size());
    return this->label_space_[idx];
  }

  typename std::vector<std::vector<LabelType>>::iterator begin() {
    return this->label_space_.begin();
  }

  typename std::vector<std::vector<LabelType>>::iterator end() {
    return this->label_space_.end();
  }

  typename std::vector<std::vector<LabelType>>::const_iterator begin() const {
    return this->label_space_.cbegin();
  }

  typename std::vector<std::vector<LabelType>>::const_iterator end() const {
    return this->label_space_.cend();
  }
};