#include "AlphaMattingProposalGenerator.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>

#include "cv_utils.h"

using namespace std;
using namespace cv;

using namespace cv_utils;

// AlphaMattingProposalGenerator::AlphaMattingProposalGenerator(const cv::Mat
// &image, const vector<bool> &source_mask, const vector<bool> &target_mask) :
// source_image_(image), source_mask_(ImageMask(source_mask, image.cols,
// image.rows)), target_mask_(ImageMask(target_mask, image.cols, image.rows)),
// IMAGE_WIDTH_(image.cols), IMAGE_HEIGHT_(image.rows)
//{
//}

AlphaMattingProposalGenerator::AlphaMattingProposalGenerator(
    const cv::Mat &image, const ImageMask &foreground_mask,
    const ImageMask &background_mask, const long seed)
    : image_(image), foreground_mask_(foreground_mask),
      background_mask_(background_mask), IMAGE_WIDTH_(image.cols),
      IMAGE_HEIGHT_(image.rows), NUM_SAMPLED_NEIGHBOR_PIXELS_(4),
      NUM_SAMPLED_REPRESENTATIVE_PIXELS_(2),
      NUM_SAMPLED_SIMILAR_COLOR_PIXELS_(2), gen(seed), dist() {
  //  foreground_mask_.dilate();
  // background_mask_.dilate();
  calcRepresentativeLabels();
  findNearestColors();
}

void AlphaMattingProposalGenerator::setCurrentSolutionCosts(
    const vector<double> &current_solution_costs) {
  current_solution_costs_ = current_solution_costs;
}

void AlphaMattingProposalGenerator::getProposals(
    LABELSPACE &proposals, const LABELSPACE &current_solution, const int N) {

  proposals.assign(IMAGE_WIDTH_ * IMAGE_HEIGHT_);
  for (int __i = 0; __i < N; ++__i) {
    LABELSPACE pixel_labels;
    vector<long> representative_labels;
    for (int i = 0; i < NUM_SAMPLED_REPRESENTATIVE_PIXELS_; i++) {
      int proposal_foreground_pixel = representative_foreground_pixels_
          [dist(gen) % representative_foreground_pixels_.size()];
      int proposal_background_pixel = representative_background_pixels_
          [dist(gen) % representative_background_pixels_.size()];
      representative_labels.push_back(
          static_cast<long>(proposal_foreground_pixel) *
              (IMAGE_WIDTH_ * IMAGE_HEIGHT_) +
          proposal_background_pixel);
    }

    pixel_labels.assign(IMAGE_WIDTH_ * IMAGE_HEIGHT_);
    for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++) {
      if (foreground_mask_.at(pixel) || background_mask_.at(pixel)) {
        pixel_labels[pixel].push_back(
            static_cast<long>(pixel) * (IMAGE_WIDTH_ * IMAGE_HEIGHT_) + pixel);
        continue;
      }
      vector<long> labels;
      long current_solution_label = current_solution(pixel, 0);
      CHECK_GE(current_solution_label, 0)
          << "current label less than 0: " << pixel << endl;

      labels.push_back(current_solution_label);
      int current_solution_foreground_pixel =
          current_solution_label / (IMAGE_WIDTH_ * IMAGE_HEIGHT_);
      int current_solution_background_pixel =
          current_solution_label % (IMAGE_WIDTH_ * IMAGE_HEIGHT_);

      int radius = max(IMAGE_WIDTH_, IMAGE_HEIGHT_);
      int num_attempts = 0;
      while (radius > 0) {
        int proposal_foreground_x =
            max(min(current_solution_foreground_pixel % IMAGE_WIDTH_ +
                        (dist(gen) % (radius * 2 + 1) - radius),
                    IMAGE_WIDTH_ - 1),
                0);
        int proposal_foreground_y =
            max(min(current_solution_foreground_pixel / IMAGE_WIDTH_ +
                        (dist(gen) % (radius * 2 + 1) - radius),
                    IMAGE_HEIGHT_ - 1),
                0);
        int proposal_background_x =
            max(min(current_solution_background_pixel % IMAGE_WIDTH_ +
                        (dist(gen) % (radius * 2 + 1) - radius),
                    IMAGE_WIDTH_ - 1),
                0);
        int proposal_background_y =
            max(min(current_solution_background_pixel / IMAGE_WIDTH_ +
                        (dist(gen) % (radius * 2 + 1) - radius),
                    IMAGE_HEIGHT_ - 1),
                0);
        int proposal_foreground_pixel =
            proposal_foreground_y * IMAGE_WIDTH_ + proposal_foreground_x;
        int proposal_background_pixel =
            proposal_background_y * IMAGE_WIDTH_ + proposal_background_x;
        if (foreground_mask_.at(proposal_foreground_pixel) == true &&
            background_mask_.at(proposal_background_pixel) == true)
          labels.push_back(static_cast<long>(proposal_foreground_pixel) *
                               (IMAGE_WIDTH_ * IMAGE_HEIGHT_) +
                           proposal_background_pixel);
        else if (foreground_mask_.at(proposal_foreground_pixel) == true)
          labels.push_back(static_cast<long>(proposal_foreground_pixel) *
                               (IMAGE_WIDTH_ * IMAGE_HEIGHT_) +
                           current_solution_background_pixel);
        else if (background_mask_.at(proposal_background_pixel) == true)
          labels.push_back(
              static_cast<long>(current_solution_foreground_pixel) *
                  (IMAGE_WIDTH_ * IMAGE_HEIGHT_) +
              proposal_background_pixel);

        radius /= 2;
      }

      // vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH_,
      // IMAGE_HEIGHT_, 4);
      vector<int> possible_neighbor_pixels = pixel_neighbors_->at(pixel);
      vector<int> neighbor_pixels;
      for (int i = 0; i < NUM_SAMPLED_NEIGHBOR_PIXELS_; i++)
        neighbor_pixels.push_back(
            possible_neighbor_pixels[dist(gen) %
                                     possible_neighbor_pixels.size()]);

      for (vector<int>::const_iterator neighbor_pixel_it =
               neighbor_pixels.begin();
           neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
        if (foreground_mask_.at(*neighbor_pixel_it) ||
            background_mask_.at(*neighbor_pixel_it))
          continue;
        long neighbor_pixel_current_solution_label =
            current_solution(*neighbor_pixel_it, 0);
        labels.push_back(neighbor_pixel_current_solution_label);
        // int neighbor_pixel_proposal_foreground_pixel = pixel -
        // *neighbor_pixel_it + neighbor_pixel_current_solution_label %
        // (IMAGE_WIDTH_ * IMAGE_HEIGHT_);
        // int neighbor_pixel_proposal_background_pixel = pixel -
        // *neighbor_pixel_it + neighbor_pixel_current_solution_label /
        // (IMAGE_WIDTH_ * IMAGE_HEIGHT_);

        // if (source_mask_.at(neighbor_pixel_proposal_label))
        //   labels.push_back(neighbor_pixel_proposal_label);
        // else
        //   labels.push_back(neighbor_pixel_current_solution_label);
      }
      labels.insert(labels.end(), representative_labels.begin(),
                    representative_labels.end());

      // if (pixel == 6468) {
      //   for (vector<int>::const_iterator label_it = labels.begin(); label_it
      //   !=
      //   labels.end(); label_it++)
      //  cout << *label_it / (IMAGE_WIDTH_ * IMAGE_HEIGHT_) << '\t' <<
      //  *label_it
      // % (IMAGE_WIDTH_ * IMAGE_HEIGHT_) << endl;
      //   cout << foreground_mask_.at(6468) << '\t' <<
      //   background_mask_.at(6468)
      //   << '\t' << foreground_mask_.at(7270) << '\t' <<
      //   background_mask_.at(7270) << endl;
      //   exit(1);
      // }

      vector<int> similar_color_pixels = histo_pixels_[pixel_histo_map_[pixel]];
      if (similar_color_pixels.size() > 0) {
        vector<int> similar_color_pixels(NUM_SAMPLED_SIMILAR_COLOR_PIXELS_);

        for (int sample_index = 0;
             sample_index < NUM_SAMPLED_SIMILAR_COLOR_PIXELS_; sample_index++)
          similar_color_pixels[sample_index] =
              similar_color_pixels[dist(gen) % similar_color_pixels.size()];
        for (vector<int>::const_iterator similar_color_pixel_it =
                 similar_color_pixels.begin();
             similar_color_pixel_it != similar_color_pixels.end();
             similar_color_pixel_it++) {
          if (foreground_mask_.at(*similar_color_pixel_it))
            labels.push_back(static_cast<long>(*similar_color_pixel_it) *
                                 (IMAGE_WIDTH_ * IMAGE_HEIGHT_) +
                             current_solution_background_pixel);
          else if (background_mask_.at(*similar_color_pixel_it))
            labels.push_back(
                static_cast<long>(current_solution_foreground_pixel) *
                    (IMAGE_WIDTH_ * IMAGE_HEIGHT_) +
                *similar_color_pixel_it);
          else
            labels.push_back(current_solution(*similar_color_pixel_it, 0));
        }
      }

      sort(labels.begin(), labels.end());
      labels.erase(unique(labels.begin(), labels.end()), labels.end());
      pixel_labels[pixel] = labels;
    }

    proposals.unionSpace(pixel_labels);
  }
}

void AlphaMattingProposalGenerator::calcRepresentativeLabels() {
  // const int NUM_REPRESENTATIVE_FOREGROUND_PIXELS =
  // sqrt(NUM_REPRESENTATIVE_LABELS_);
  // const int NUM_REPRESENTATIVE_BACKGROUND_PIXELS =
  // sqrt(NUM_REPRESENTATIVE_LABELS_);

  Mat blurred_image;
  blur(image_, blurred_image, Size(15, 15));

  Mat samples = Mat::zeros(IMAGE_WIDTH_ * IMAGE_HEIGHT_, 3, CV_32F);
  Mat labels, centers;
  for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++) {
    Vec3b color =
        blurred_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_);
    // p.at<float>(i,0) = (i/src.cols) / src.rows;
    //   p.at<float>(i,1) = (i%src.cols) / src.cols;
    for (int c = 0; c < 3; c++)
      samples.at<float>(pixel, c) = 1.0 * color[c] / 255;
  }

  int K = 20;
  cv::kmeans(samples, K, labels,
             TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 3,
             KMEANS_PP_CENTERS, centers);

  map<int, vector<int>> foreground_clusters;
  map<int, vector<int>> background_clusters;
  for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++) {
    int label = labels.at<int>(0, pixel);
    if (foreground_mask_.at(pixel))
      foreground_clusters[label].push_back(pixel);
    if (background_mask_.at(pixel))
      background_clusters[label].push_back(pixel);
  }
  // sort(foreground_clusters.begin(), foreground_clusters.end(), [](const
  // vector<int> &a, const vector<int> &b) { return a.size() >= b.size(); });
  // sort(background_clusters.begin(), background_clusters.end(), [](const
  // vector<int> &a, const vector<int> &b) { return a.size() >= b.size(); });

  representative_foreground_pixels_.clear();
  for (map<int, vector<int>>::const_iterator cluster_it =
           foreground_clusters.begin();
       cluster_it != foreground_clusters.end(); cluster_it++)
    representative_foreground_pixels_.push_back(
        cluster_it->second[dist(gen) % cluster_it->second.size()]);
  representative_background_pixels_.clear();
  for (map<int, vector<int>>::const_iterator cluster_it =
           background_clusters.begin();
       cluster_it != background_clusters.end(); cluster_it++)
    representative_background_pixels_.push_back(
        cluster_it->second[dist(gen) % cluster_it->second.size()]);
  if (representative_foreground_pixels_.size() == 0) {
    vector<int> foreground_pixels = foreground_mask_.getPixels();
    representative_foreground_pixels_.push_back(
        foreground_pixels[dist(gen) % foreground_pixels.size()]);
  }
  if (representative_background_pixels_.size() == 0) {
    vector<int> background_pixels = background_mask_.getPixels();
    representative_background_pixels_.push_back(
        background_pixels[dist(gen) % background_pixels.size()]);
  }

  Mat clustered_image = Mat(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
  map<int, Vec3b> color_table;
  for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++) {
    int label = labels.at<int>(0, pixel);
    if (color_table.count(label) == 0)
      color_table[label] =
          Vec3b(dist(gen) % 256, dist(gen) % 256, dist(gen) % 256);
    clustered_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) =
        color_table[label];
  }

  imwrite("Test/clustered_image.bmp", clustered_image);
}

void AlphaMattingProposalGenerator::setNeighbors(
    const std::shared_ptr<const std::vector<std::vector<int>>>
        pixel_neighbors) {
  pixel_neighbors_ = pixel_neighbors;
}

void AlphaMattingProposalGenerator::findNearestColors() {
  const int SIMILAR_COLOR_THRESHOLD = 10;
  const int NUM_HISTOS_PER_CHANNEL = ceil(255.0 / SIMILAR_COLOR_THRESHOLD);

  pixel_histo_map_.assign(IMAGE_WIDTH_ * IMAGE_HEIGHT_, 0);
  histo_pixels_.assign(pow(NUM_HISTOS_PER_CHANNEL, 3), vector<int>());
  for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++) {
    Vec3b color = image_.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_);
    int histo_index = 0;
    for (int c = 0; c < 3; c++)
      histo_index = histo_index * NUM_HISTOS_PER_CHANNEL +
                    color[c] / SIMILAR_COLOR_THRESHOLD;
    pixel_histo_map_[pixel] = histo_index;
    histo_pixels_[histo_index].push_back(pixel);
  }
}

// vector<vector<pair<double, long> > >
// AlphaMattingProposalGenerator::findPropagationLabels()
// {
//   vector<vector<pair<double, long> > >
//   propagation_cost_label_pairs(NUM_PIXELS_);
//   if (propagation_direction_) {
//     for (int step = 0; step <= max(IMAGE_WIDTH_, IMAGE_HEIGHT_) * 2; step++)
//     {
//       for (int i = 0; i < max(IMAGE_WIDTH_, IMAGE_HEIGHT_) * 2; i++) {
//  int x = IMAGE_WIDTH_ - 1 - (step - 1 - i);
//  int y = IMAGE_HEIGHT_ - 1 - i;
//  if (x < 0 || x >= IMAGE_WIDTH_ || y < 0 || y >= IMAGE_HEIGHT_)
//    continue;
//  int pixel = y * IMAGE_WIDTH_ + x;
//  if (foreground_mask_.at(pixel) || background_mask_.at(pixel))
//           continue;
//  vector<pair<double, long> >
// cost_label_pairs.push_back(make_pair(current_solution_costs_[pixel],
// current_solution_[pixel]));
//  if (x < IMAGE_WIDTH_ - 1 && foreground_mask_.at(pixel + 1) == false &&
// background_mask_.at(pixel + 1) == false)
//    cost_label_pairs.insert(propagation_cost_label_pairs[pixel].end(),
// propagation_cost_label_pairs[pixel + 1].begin(),
// propagation_cost_label_pairs[pixel + 1].end());
//  if (y < IMAGE_HEIGHT_ - 1 && foreground_mask_.at(pixel + IMAGE_WIDTH_)
// == false && background_mask_.at(pixel + IMAGE_WIDTH_) == false)
//    cost_label_pairs.insert(propagation_cost_label_pairs[pixel].end(),
// propagation_cost_label_pairs[pixel + IMAGE_WIDTH_].begin(),
// propagation_cost_label_pairs[pixel + IMAGE_WIDTH_].end());
//  sort(cost_label_pairs.begin(), cost_label_pairs.end());
//  cost_label_pairs.erase(cost_label_pairs.begin() +
// NUM_PROPAGATION_LABELS_, cost_label_pairs.end());
//  propagation_cost_label_pairs[pixel] = cost_label_pairs;
//       }
//     }
//   } else {
//     for (int step = 0; step <= max(IMAGE_WIDTH_, IMAGE_HEIGHT_) * 2; step++)
//     {
//       for (int i = 0; i < max(IMAGE_WIDTH_, IMAGE_HEIGHT_) * 2; i++) {
//         int x = step - 1 - i;
//         int y = i;
//         if (x < 0 || x >= IMAGE_WIDTH_ || y < 0 || y >= IMAGE_HEIGHT_)
//           continue;
//         int pixel = y * IMAGE_WIDTH_ + x;
//         if (foreground_mask_.at(pixel) || background_mask_.at(pixel))
//           continue;
//         vector<pair<double, long> >
//         cost_label_pairs.push_back(make_pair(current_solution_costs_[pixel],
//         current_solution_[pixel]));
//         if (x > 0 && foreground_mask_.at(pixel - 1) == false &&
//         background_mask_.at(pixel - 1) == false)
//           cost_label_pairs.insert(propagation_cost_label_pairs[pixel].end(),
//           propagation_cost_label_pairs[pixel - 1].begin(),
//           propagation_cost_label_pairs[pixel - 1].end());
//         if (y > 0 && foreground_mask_.at(pixel - IMAGE_WIDTH_) == false &&
//         background_mask_.at(pixel - IMAGE_WIDTH_) == false)
//           cost_label_pairs.insert(propagation_cost_label_pairs[pixel].end(),
//           propagation_cost_label_pairs[pixel - IMAGE_WIDTH_].begin(),
//           propagation_cost_label_pairs[pixel - IMAGE_WIDTH_].end());
//         sort(cost_label_pairs.begin(), cost_label_pairs.end());
//         cost_label_pairs.erase(cost_label_pairs.begin() +
//         NUM_PROPAGATION_LABELS_, cost_label_pairs.end());
//  propagation_cost_label_pairs[pixel] = cost_label_pairs;
//       }
//     }
//   }
//   return propagation_cost_label_pairs;
// }
