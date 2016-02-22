#ifndef IMAGE_COMPLETION_COST_FUNCTOR_H__
#define IMAGE_COMPLETION_COST_FUNCTOR_H__

#include <opencv2/core/core.hpp>
#include <vector>
#include <map>
#include <string>

#include "cv_utils.h"
#include "CostFunctor.h"

//class cv_utils::ImageMask;

class AlphaMattingCostFunctor : public CostFunctor
{
 public:
  AlphaMattingCostFunctor(const cv::Mat &image, const std::vector<bool> &foreground_mask, const std::vector<bool> &background_mask);
  AlphaMattingCostFunctor(const cv::Mat &image, const cv_utils::ImageMask &foreground_mask, const cv_utils::ImageMask &background_mask, const std::string image_identifier);
  
  //virtual void setCurrentSolution(const std::vector<int> &current_solution);
  double calcAlpha(const int pixel, const long label) const;
  
  virtual double operator()(const int node_index, const long label) const;
  virtual double operator()(const int node_index_1, const int node_index_2, const long label_1, const long label_2) const;
  
  std::vector<std::vector<int> > getPixelNeighbors() const;
  
 private:
  const cv::Mat image_;
  std::vector<std::vector<int> > pixel_neighbors_;
  std::vector<std::map<int, double> > pixel_neighbor_weights_;
  
  const cv_utils::ImageMask foreground_mask_;
  const cv_utils::ImageMask background_mask_;
  const std::string image_identifier_;
  
  const int IMAGE_WIDTH_;
  const int IMAGE_HEIGHT_;
  
  const int NEIGHBOR_WINDOW_SIZE_;
  const int NUM_NEIGHBORS_;
  
  const double DATA_TERM_WEIGHT_;
  const double SMOOTHNESS_TERM_WEIGHT_;
  
  std::vector<double> foreground_distance_map_;
  std::vector<double> background_distance_map_;
  
  
  void calcNeighborsInfo();
  void calcNeighborsInfoGeodesicDistance();
  void calcDistanceMaps();
};

#endif
