#include "AlphaMattingCostFunctor.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fmt/format.h>
#include <fstream>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>
#include <string.h>

using namespace cv;
using namespace std;
using namespace cv_utils;
using namespace fmt::literals;

AlphaMattingCostFunctor::AlphaMattingCostFunctor(
    const cv::Mat &image, const ImageMask &foreground_mask,
    const ImageMask &background_mask, const string image_identifier)
    : image_(image.clone()), foreground_mask_(foreground_mask),
      background_mask_(background_mask), IMAGE_WIDTH_(image.cols),
      IMAGE_HEIGHT_(image.rows), NEIGHBOR_WINDOW_SIZE_(5), NUM_NEIGHBORS_(9),
      DATA_TERM_WEIGHT_(1.0), SMOOTHNESS_TERM_WEIGHT_(1),
      image_identifier_(image_identifier),
      pixel_neighbors_(std::make_shared<std::vector<std::vector<int>>>()) {
  calcNeighborsInfo();
  calcDistanceMaps();
}

double AlphaMattingCostFunctor::operator()(const int pixel,
                                           const long label) const {
  const int foreground_pixel = label / (IMAGE_WIDTH_ * IMAGE_HEIGHT_);
  const int background_pixel = label % (IMAGE_WIDTH_ * IMAGE_HEIGHT_);
  Vec3b foreground_color = image_.at<Vec3b>(foreground_pixel / IMAGE_WIDTH_,
                                            foreground_pixel % IMAGE_WIDTH_);
  Vec3b background_color = image_.at<Vec3b>(background_pixel / IMAGE_WIDTH_,
                                            background_pixel % IMAGE_WIDTH_);
  Vec3b color = image_.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_);
  double alpha = calcAlpha(pixel, label);
  double data_cost = 0;
  for (int c = 0; c < 3; c++)
    data_cost += pow(color[c] - (alpha * foreground_color[c] +
                                 (1 - alpha) * background_color[c]),
                     2);
  if (data_cost < 0 || data_cost > pow(255.0, 2) * 3 ||
      std::isnan(static_cast<double>(data_cost))) {
    cout << alpha << '\t' << data_cost << endl;
    exit(1);
  }

  if (foreground_mask_.at(pixel) == false &&
      background_mask_.at(pixel) == false) {
    const double distance_weight = 1;
    data_cost +=
        sqrt(pow(foreground_pixel % IMAGE_WIDTH_ - pixel % IMAGE_WIDTH_, 2) +
             pow(foreground_pixel / IMAGE_WIDTH_ - pixel / IMAGE_WIDTH_, 2)) /
            foreground_distance_map_[pixel] +
        sqrt(pow(background_pixel % IMAGE_WIDTH_ - pixel % IMAGE_WIDTH_, 2) +
             pow(background_pixel / IMAGE_WIDTH_ - pixel / IMAGE_WIDTH_, 2)) /
            background_distance_map_[pixel];
  }

  return data_cost * DATA_TERM_WEIGHT_;
}

double AlphaMattingCostFunctor::operator()(const int pixel_1, const int pixel_2,
                                           const long label_1,
                                           const long label_2) const {
  assert(pixel_1 < pixel_2);
  double alpha_1 = calcAlpha(pixel_1, label_1);
  double alpha_2 = calcAlpha(pixel_2, label_2);
  return abs(alpha_1 - alpha_2) * SMOOTHNESS_TERM_WEIGHT_ *
         pixel_neighbor_weights_[min(pixel_1, pixel_2)].at(
             max(pixel_1, pixel_2));
}

double AlphaMattingCostFunctor::calcAlpha(const int pixel,
                                          const long label) const {
  if (foreground_mask_.at(pixel))
    return 1.0;
  if (background_mask_.at(pixel))
    return 0.0;
  const int foreground_pixel = label / (IMAGE_WIDTH_ * IMAGE_HEIGHT_);
  const int background_pixel = label % (IMAGE_WIDTH_ * IMAGE_HEIGHT_);
  Vec3b foreground_color = image_.at<Vec3b>(foreground_pixel / IMAGE_WIDTH_,
                                            foreground_pixel % IMAGE_WIDTH_);
  Vec3b background_color = image_.at<Vec3b>(background_pixel / IMAGE_WIDTH_,
                                            background_pixel % IMAGE_WIDTH_);
  Vec3b color = image_.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_);
  double alpha_numerator = 0, alpha_denominator = 0;
  for (int c = 0; c < 3; c++) {
    alpha_numerator += (color[c] - background_color[c]) *
                       (foreground_color[c] - background_color[c]);
    alpha_denominator += pow(foreground_color[c] - background_color[c], 2);
  }
  double alpha = abs(alpha_denominator) > 0.000001
                     ? alpha_numerator / alpha_denominator
                     : 0.5;
  alpha = max(min(alpha, 1.0), 0.0);
  if (alpha < 0 || alpha > 1 || std::isnan(static_cast<double>(alpha))) {
    cout << foreground_color << '\t' << background_color << endl;
    cout << foreground_pixel << '\t' << background_pixel << endl;
    cout << pixel << '\t' << alpha << endl;
    exit(1);
  }
  return alpha;
}

// void AlphaMattingCostFunctor::calcNeighborsInfo()
// {
//   pixel_neighbors_->assign(IMAGE_WIDTH_ * IMAGE_HEIGHT_, vector<int>());
//   pixel_neighbor_weights_.assign(IMAGE_WIDTH_ * IMAGE_HEIGHT_ * IMAGE_WIDTH_
//   * IMAGE_HEIGHT_, 0);
//   for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++) {
//     if (pixel % 100 == 0)
//       cout << pixel << endl;
//     int x = pixel % IMAGE_WIDTH_;
//     int y = pixel / IMAGE_WIDTH_;
//     Vec3b color_1 = image_.at<Vec3b>(y, x);
//     for (int delta_x = -NEIGHBOR_WINDOW_SIZE_; delta_x <=
//     NEIGHBOR_WINDOW_SIZE_; delta_x++) {
//       for (int delta_y = -NEIGHBOR_WINDOW_SIZE_; delta_y <=
//       NEIGHBOR_WINDOW_SIZE_; delta_y++) {
//  if (x + delta_x >= 0 && x + delta_x < IMAGE_WIDTH_ && y + delta_y >= 0
// && y + delta_y < IMAGE_HEIGHT_) {
//    pixel_neighbors_->at(pixel).push_back((y + delta_y) * IMAGE_WIDTH_ + (x +
// delta_x));

//    double weight = 0;
//    Vec3b color_2 = image_.at<Vec3b>(y + delta_y, x + delta_x);
//           int window_min_x = max(max(x, x + delta_x) - NEIGHBOR_WINDOW_SIZE_,
//           0);
//    int window_max_x = min(min(x, x + delta_x) + NEIGHBOR_WINDOW_SIZE_,
// IMAGE_WIDTH_ - 1);
//           int window_min_y = max(max(y, y + delta_y) - NEIGHBOR_WINDOW_SIZE_,
//           0);
//           int window_max_y = min(min(y, y + delta_y) + NEIGHBOR_WINDOW_SIZE_,
//           IMAGE_HEIGHT_ - 1);
//    for (int window_start_x = window_min_x; window_start_x <= window_max_x
// - NEIGHBOR_WINDOW_SIZE_; window_start_x++) {
//      for (int window_start_y = window_min_y; window_start_y <=
// window_max_y - NEIGHBOR_WINDOW_SIZE_; window_start_y++) {
//        vector<vector<double> > colors;
//        for (int window_x = window_start_x; window_x <= window_start_x +
// NEIGHBOR_WINDOW_SIZE_; window_x++) {
//    for (int window_y = window_start_y; window_y <= window_start_y +
// NEIGHBOR_WINDOW_SIZE_; window_y++) {
//      Vec3b color = image_.at<Vec3b>(window_y, window_x);
//      vector<double> color_vec(3);
//      for (int c = 0; c < 3; c++)
//                     color_vec[c] = color[c];
//      colors.push_back(color_vec);
//    }
//        }
//        vector<double> color_mean;
//        vector<vector<double> > color_var;
//        calcMeanAndSVar(colors, color_mean, color_var);
//        for (int c = 0; c < 3; c++)
//    color_var[c][c] += 0.000001;
//        vector<vector<double> > color_inverse_var =
// calcInverse(color_var);
//        double window_weight = 0;
//        for (int c = 0; c < 3; c++)
//    for (int d = 0; d < 3; d++)
//      window_weight += (color_1[c] - color_mean[c]) *
// color_inverse_var[c][d] * (color_2[d] - color_mean[d]);
//        weight += 1 + window_weight;
//      }
//    }
//    pixel_neighbor_weights_[pixel * (IMAGE_WIDTH_ * IMAGE_HEIGHT_) + ((y +
// delta_y) * IMAGE_WIDTH_ + (x + delta_x))] = weight;
//  }
//       }
//     }
//   }
// }

void AlphaMattingCostFunctor::calcNeighborsInfo() {
  stringstream neighbor_info_filename;
  neighbor_info_filename << "Cache/{}_neighbor_info"_format(image_identifier_);
  ifstream neighbor_info_in_str(neighbor_info_filename.str());
  if (neighbor_info_in_str && false) {
    pixel_neighbors_->assign(IMAGE_WIDTH_ * IMAGE_HEIGHT_, vector<int>());
    pixel_neighbor_weights_.assign(IMAGE_WIDTH_ * IMAGE_HEIGHT_,
                                   map<int, double>());
    for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++) {
      int num_neighbors;
      int pixel_temp;
      neighbor_info_in_str >> pixel_temp >> num_neighbors;
      if (num_neighbors != 0) {
        if (pixel_temp != pixel) {
          fmt::print("{}\t{}\t{}", pixel, pixel_temp, num_neighbors);
          exit(1);
        }
      }
      for (int i = 0; i < num_neighbors; i++) {
        int neighbor_pixel;
        double weight;
        neighbor_info_in_str >> neighbor_pixel >> weight;
        // cout << neighbor_pixel << '\t' << weight << endl;
        pixel_neighbors_->at(pixel).push_back(neighbor_pixel);
        pixel_neighbor_weights_[pixel][neighbor_pixel] = weight;
      }
    }

    neighbor_info_in_str.close();

    // double sum = 0;
    // for (map<int, double>::const_iterator neighbor_pixel_it =
    // pixel_neighbor_weights_[37570].begin(); neighbor_pixel_it !=
    // pixel_neighbor_weights_[37570].end(); neighbor_pixel_it++) {
    //   cout << neighbor_pixel_it->first % IMAGE_WIDTH_<< '\t' <<
    //   neighbor_pixel_it->first / IMAGE_WIDTH_ << '\t' <<
    //   neighbor_pixel_it->second << endl;
    //   sum += neighbor_pixel_it->second;
    // }
    // cout << sum << endl;
    // exit(1);

    return;
  }

  pixel_neighbors_->assign(IMAGE_WIDTH_ * IMAGE_HEIGHT_, vector<int>());
  pixel_neighbor_weights_.assign(IMAGE_WIDTH_ * IMAGE_HEIGHT_,
                                 map<int, double>());

  ImageMask unknown_mask = ImageMask(true, IMAGE_WIDTH_, IMAGE_HEIGHT_) -
                           foreground_mask_ - background_mask_;
  // for (int c = 0; c < NEIGHBOR_WINDOW_SIZE_ - 1; c++)
  // unknown_mask.dilate();

  const int NUM_PIXELS = IMAGE_WIDTH_ * IMAGE_HEIGHT_;
  vector<vector<double>> guidance_image_values(3, vector<double>(NUM_PIXELS));
  for (int y = 0; y < IMAGE_HEIGHT_; y++) {
    for (int x = 0; x < IMAGE_WIDTH_; x++) {
      int pixel = y * IMAGE_WIDTH_ + x;
      Vec3b guidance_image_color = image_.at<Vec3b>(y, x);
      for (int c = 0; c < 3; c++) {
        guidance_image_values[c][pixel] = 1.0 * guidance_image_color[c] / 256;
      }
    }
  }

  vector<vector<double>> guidance_image_means;
  vector<vector<double>> guidance_image_vars;
  calcWindowMeansAndVars(guidance_image_values, IMAGE_WIDTH_, IMAGE_HEIGHT_,
                         NEIGHBOR_WINDOW_SIZE_, guidance_image_means,
                         guidance_image_vars);

  double epsilon = 0.00001;
  for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++) {
    vector<int> unknown_window_pixels =
        unknown_mask.findMaskWindowPixels(pixel, NEIGHBOR_WINDOW_SIZE_);
    vector<int> window_pixels = findWindowPixels(
        pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_, NEIGHBOR_WINDOW_SIZE_);
    if (unknown_window_pixels.size() == 0)
      continue;
    vector<vector<double>> guidance_image_var(3, vector<double>(3));
    for (int c_1 = 0; c_1 < 3; c_1++)
      for (int c_2 = 0; c_2 < 3; c_2++)
        guidance_image_var[c_1][c_2] =
            guidance_image_vars[c_1 * 3 + c_2][pixel] +
            epsilon / 9 * (c_1 == c_2);
    vector<double> guidance_image_mean(3);
    for (int c = 0; c < 3; c++)
      guidance_image_mean[c] = guidance_image_means[c][pixel];
    vector<vector<double>> guidance_image_var_inverse =
        calcInverse(guidance_image_var);
    for (vector<int>::const_iterator window_pixel_it = window_pixels.begin();
         window_pixel_it != window_pixels.end(); window_pixel_it++) {
      for (vector<int>::const_iterator unknown_window_pixel_it =
               unknown_window_pixels.begin();
           unknown_window_pixel_it != unknown_window_pixels.end();
           unknown_window_pixel_it++) {
        if (*unknown_window_pixel_it == *window_pixel_it)
          continue;
        vector<double> color_1(3);
        for (int c = 0; c < 3; c++)
          color_1[c] = guidance_image_values[c][*window_pixel_it];
        vector<double> color_2(3);
        for (int c = 0; c < 3; c++)
          color_2[c] = guidance_image_values[c][*unknown_window_pixel_it];

        double weight = 0;
        for (int c_1 = 0; c_1 < 3; c_1++)
          for (int c_2 = 0; c_2 < 3; c_2++)
            weight += (color_1[c_1] - guidance_image_mean[c_1]) *
                      guidance_image_var_inverse[c_1][c_2] *
                      (color_2[c_2] - guidance_image_mean[c_2]);
        weight = (weight + 1) / pow(NEIGHBOR_WINDOW_SIZE_, 4);

        // if (weight < 0) {
        //   cout << pixel << '\t' << window_pixels.size() << endl;
        //   for (int c_1 = 0; c_1 < 3; c_1++)
        //     for (int c_2 = 0; c_2 < 3; c_2++)
        //       cout << guidance_image_var[c_1][c_2] << '\t' <<
        //       guidance_image_var_inverse[c_1][c_2] << endl;
        //   cout << weight << endl;
        //   for (int c = 0; c < 3; c++)
        //     cout << guidance_image_mean[c] << '\t' << color_1[c]  << '\t' <<
        //     color_2[c] << endl;
        //    exit(1);
        // }

        // if (min(*window_pixel_it, *unknown_window_pixel_it) == 19586) {
        //   cout << pixel << '\t' << window_pixels.size() << endl;
        //   for (int c_1 = 0; c_1 < 3; c_1++)
        //     for (int c_2 = 0; c_2 < 3; c_2++)
        //       cout << guidance_image_var[c_1][c_2] << '\t' <<
        //       guidance_image_var_inverse[c_1][c_2] << endl;
        //   cout << weight << endl;
        //   //exit(1);
        // }

        // if (abs(*unknown_window_pixel_it % IMAGE_WIDTH_ - *window_pixel_it %
        // IMAGE_WIDTH_) > 1 || abs(*unknown_window_pixel_it / IMAGE_WIDTH_ -
        // *window_pixel_it / IMAGE_WIDTH_) > 1)
        // continue;
        pixel_neighbor_weights_[min(*window_pixel_it, *unknown_window_pixel_it)]
                               [max(*window_pixel_it,
                                    *unknown_window_pixel_it)] += weight;
        // pixel_neighbor_weights_[max(*window_pixel_it,
        // *unknown_window_pixel_it)][min(*window_pixel_it,
        // *unknown_window_pixel_it)] += weight;
      }
    }
  }

  // vector<map<int, double> > half_window_pixel_neighbor_weights(IMAGE_WIDTH_ *
  // IMAGE_HEIGHT_);
  // for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++)
  //   for (map<int, double>::const_iterator neighbor_pixel_it =
  //   pixel_neighbor_weights_[pixel].begin(); neighbor_pixel_it !=
  //   pixel_neighbor_weights_[pixel].end(); neighbor_pixel_it++)
  //     if (neighbor_pixel_it->first > pixel)
  //  half_window_pixel_neighbor_weights[pixel][neighbor_pixel_it->first] =
  // neighbor_pixel_it->second / 2;
  // pixel_neighbor_weights_ = half_window_pixel_neighbor_weights;

  // double sum = 0;
  // for (map<int, double>::const_iterator neighbor_pixel_it =
  // pixel_neighbor_weights_[230].begin(); neighbor_pixel_it !=
  // pixel_neighbor_weights_[230].end(); neighbor_pixel_it++) {
  //   cout << neighbor_pixel_it->first % IMAGE_WIDTH_ << '\t' <<
  //   neighbor_pixel_it->first / IMAGE_WIDTH_ << '\t' <<
  //   neighbor_pixel_it->second << endl;
  //   sum += neighbor_pixel_it->second;
  // }
  // cout << sum << endl;
  // exit(1);

  for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++)
    for (map<int, double>::const_iterator neighbor_pixel_it =
             pixel_neighbor_weights_[pixel].begin();
         neighbor_pixel_it != pixel_neighbor_weights_[pixel].end();
         neighbor_pixel_it++)
      pixel_neighbors_->at(pixel).push_back(neighbor_pixel_it->first);

  ofstream neighbor_info_out_str(neighbor_info_filename.str());
  for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++) {
    // if (pixel_neighbors_->at(pixel).size() == 0)
    // continue;
    if (pixel_neighbors_->at(pixel).size() !=
        pixel_neighbor_weights_[pixel].size()) {
      cout << pixel << '\t' << pixel_neighbors_->at(pixel).size() << '\t'
           << pixel_neighbor_weights_[pixel].size() << endl;
      exit(1);
    }
    neighbor_info_out_str << pixel << '\t' << pixel_neighbors_->at(pixel).size()
                          << endl;
    for (map<int, double>::const_iterator neighbor_pixel_it =
             pixel_neighbor_weights_[pixel].begin();
         neighbor_pixel_it != pixel_neighbor_weights_[pixel].end();
         neighbor_pixel_it++) {
      neighbor_info_out_str << neighbor_pixel_it->first << '\t'
                            << neighbor_pixel_it->second << endl;
    }
  }
  neighbor_info_out_str.close();
}

void AlphaMattingCostFunctor::calcNeighborsInfoGeodesicDistance() {
  stringstream neighbor_info_filename;
  neighbor_info_filename << "Cache/" + image_identifier_ + "_neighbor_info";
  ifstream neighbor_info_in_str(neighbor_info_filename.str());
  if (neighbor_info_in_str) {
    pixel_neighbors_->assign(IMAGE_WIDTH_ * IMAGE_HEIGHT_, vector<int>());
    pixel_neighbor_weights_.assign(IMAGE_WIDTH_ * IMAGE_HEIGHT_,
                                   map<int, double>());
    for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++) {
      int num_neighbors;
      neighbor_info_in_str >> num_neighbors;
      for (int i = 0; i < num_neighbors; i++) {
        int neighbor_pixel;
        double weight;
        neighbor_info_in_str >> neighbor_pixel >> weight;
        pixel_neighbors_->at(pixel).push_back(neighbor_pixel);
        pixel_neighbor_weights_[pixel][neighbor_pixel] = weight;
      }
    }
    neighbor_info_in_str.close();
    return;
  }

  vector<vector<double>> distance_map(IMAGE_WIDTH_ * IMAGE_HEIGHT_);
  for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++) {
    vector<int> neighbor_pixels =
        findNeighbors(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_);
    vector<double> neighbor_distances(9, 0);
    Vec3b color = image_.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_);
    for (vector<int>::const_iterator neighbor_pixel_it =
             neighbor_pixels.begin();
         neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
      Vec3b neighbor_color = image_.at<Vec3b>(
          *neighbor_pixel_it / IMAGE_WIDTH_, *neighbor_pixel_it % IMAGE_WIDTH_);
      double color_distance = 0;
      for (int c = 0; c < 3; c++)
        color_distance += pow(color[c] - neighbor_color[c], 2);
      color_distance = sqrt(color_distance);
      neighbor_distances
          [(*neighbor_pixel_it / IMAGE_WIDTH_ - pixel / IMAGE_WIDTH_ + 1) * 3 +
           (*neighbor_pixel_it % IMAGE_WIDTH_ - pixel % IMAGE_WIDTH_ + 1)] =
              color_distance;
    }
    distance_map[pixel] = neighbor_distances;
  }

  pixel_neighbors_->assign(IMAGE_WIDTH_ * IMAGE_HEIGHT_, vector<int>());
  pixel_neighbor_weights_.assign(IMAGE_WIDTH_ * IMAGE_HEIGHT_,
                                 map<int, double>());
  vector<double> distances;
  for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++) {
    if (pixel % 100000 == 0)
      cout << pixel << endl;
    vector<pair<double, int>> distance_neighbor_pairs;
    int x = pixel % IMAGE_WIDTH_;
    int y = pixel / IMAGE_WIDTH_;
    Vec3b color_1 = image_.at<Vec3b>(y, x);
    vector<int> end_pixels;
    for (int delta_x = -(NEIGHBOR_WINDOW_SIZE_ - 1) / 2;
         delta_x <= (NEIGHBOR_WINDOW_SIZE_ - 1) / 2; delta_x++) {
      for (int delta_y = 0; delta_y <= (NEIGHBOR_WINDOW_SIZE_ - 1) / 2;
           delta_y++) {
        if (delta_y * NEIGHBOR_WINDOW_SIZE_ + delta_x > 0 && x + delta_x >= 0 &&
            x + delta_x < IMAGE_WIDTH_ && y + delta_y >= 0 &&
            y + delta_y < IMAGE_HEIGHT_)
          if ((foreground_mask_.at(pixel) == false &&
               background_mask_.at(pixel) == false) ||
              (foreground_mask_.at((y + delta_y) * IMAGE_WIDTH_ +
                                   (x + delta_x)) == false &&
               background_mask_.at((y + delta_y) * IMAGE_WIDTH_ +
                                   (x + delta_x)) == false))
            end_pixels.push_back((y + delta_y) * IMAGE_WIDTH_ + (x + delta_x));
      }
    }
    if (end_pixels.size() == 0)
      continue;

    vector<double> geodesic_distances = calcGeodesicDistances(
        distance_map, IMAGE_WIDTH_, IMAGE_HEIGHT_, pixel, end_pixels, 0);
    for (vector<int>::const_iterator end_pixel_it = end_pixels.begin();
         end_pixel_it != end_pixels.end(); end_pixel_it++) {
      double distance = geodesic_distances[end_pixel_it - end_pixels.begin()];
      distance_neighbor_pairs.push_back(make_pair(distance, *end_pixel_it));
    }
    sort(distance_neighbor_pairs.begin(), distance_neighbor_pairs.end());
    // cout << pixel << '\t' << distance_neighbor_pairs.size() << endl;
    // for (int i = 0; i < min(NUM_NEIGHBORS_,
    // static_cast<int>(distance_neighbor_pairs.size())); i++) {
    for (int i = 0; i < distance_neighbor_pairs.size(); i++) {
      if (i < NUM_NEIGHBORS_ ||
          (abs(distance_neighbor_pairs[i].second % IMAGE_WIDTH_ - x) <= 1 &&
           abs(distance_neighbor_pairs[i].second / IMAGE_WIDTH_ - y) <= 1)) {
        pixel_neighbors_->at(pixel).push_back(
            distance_neighbor_pairs[i].second);
        pixel_neighbor_weights_[pixel][distance_neighbor_pairs[i].second] =
            distance_neighbor_pairs[i].first;
        distances.push_back(distance_neighbor_pairs[i].first);
      }
    }
    // cout << pixel_neighbors_->at(pixel).size() << endl;
    // exit(1);
  }

  vector<double> distance_mean_and_svar = calcMeanAndSVar(distances);
  for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++)
    for (map<int, double>::iterator neighbor_it =
             pixel_neighbor_weights_[pixel].begin();
         neighbor_it != pixel_neighbor_weights_[pixel].end(); neighbor_it++)
      neighbor_it->second = exp(-pow(neighbor_it->second, 2) /
                                (2 * pow(distance_mean_and_svar[1], 2)));

  ofstream neighbor_info_out_str(neighbor_info_filename.str());
  for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++) {
    neighbor_info_out_str << pixel_neighbors_->at(pixel).size() << endl;
    for (map<int, double>::const_iterator neighbor_pixel_it =
             pixel_neighbor_weights_[pixel].begin();
         neighbor_pixel_it != pixel_neighbor_weights_[pixel].end();
         neighbor_pixel_it++) {
      neighbor_info_out_str << neighbor_pixel_it->first << '\t'
                            << neighbor_pixel_it->second << endl;
      ;
    }
  }
  neighbor_info_out_str.close();
}

const std::shared_ptr<const std::vector<std::vector<int>>>
AlphaMattingCostFunctor::getPixelNeighbors() const {
  return pixel_neighbors_;
}

void AlphaMattingCostFunctor::calcDistanceMaps() {
  foreground_distance_map_ = foreground_mask_.calcDistanceMapOutside();
  background_distance_map_ = background_mask_.calcDistanceMapOutside();
}
