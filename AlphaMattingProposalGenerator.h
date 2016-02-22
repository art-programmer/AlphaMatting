#ifndef ALPHA_MATTING_PROPOSAL_GENERATOR_H__
#define ALPHA_MATTING_PROPOSAL_GENERATOR_H__

#include <vector>

#include "cv_utils.h"
#include "ProposalGenerator.h"

//class cv_utils::ImageMask;

class AlphaMattingProposalGenerator : public ProposalGenerator
{
 public:
  //AlphaMattingProposalGenerator(const cv::Mat &image, const std::vector<bool> &source_mask, const std::vector<bool> &target_mask);
  AlphaMattingProposalGenerator(const cv::Mat &image, const cv_utils::ImageMask &foreground_mask, const cv_utils::ImageMask &background_mask);
  
  //void setCurrentSolution(const std::vector<int> &current_solution);
  void setNeighbors(const std::vector<std::vector<int> > &pixel_neighbors);
  
  virtual void setCurrentSolution(const std::vector<long> &current_solution);
  virtual std::vector<std::vector<long> > getProposal() const;

  void setCurrentSolutionCosts(const std::vector<double> &current_solution_costs);
  
 private:
  const cv::Mat image_;
  cv_utils::ImageMask foreground_mask_;
  cv_utils::ImageMask background_mask_;
  
  const int IMAGE_WIDTH_;
  const int IMAGE_HEIGHT_;
  
  const int NUM_SAMPLED_NEIGHBOR_PIXELS_;
  const int NUM_SAMPLED_REPRESENTATIVE_PIXELS_;
  const int NUM_SAMPLED_SIMILAR_COLOR_PIXELS_;
  
  std::vector<double> current_solution_costs_;
  
  std::vector<int> representative_foreground_pixels_;
  std::vector<int> representative_background_pixels_;
  
  std::vector<std::vector<int> > pixel_neighbors_;
  
  std::vector<int> pixel_histo_map_;
  std::vector<std::vector<int> > histo_pixels_;
  
  void calcRepresentativeLabels();
  void findNearestColors();
};

#endif
