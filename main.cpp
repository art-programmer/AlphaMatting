#include "AlphaMattingCostFunctor.h"
#include "AlphaMattingProposalGenerator.h"
#include "FusionSpaceSolver.h"
#include "ParallelFusionPipeline.h"
#include "cv_utils.h"

#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <spdlog/spdlog.h>

using namespace std;
using namespace cv;
using namespace cv_utils;
using namespace fmt::literals;

DEFINE_string(base_dir, "/home/erik/Projects/ParallelFusion/AlphaMatting",
              "Base directory");
DEFINE_string(img_name, "GT24", "Image name");
DEFINE_int32(num_threads, 4, "Number of threads");
DEFINE_int32(num_proposals, 1, "Number of proposals per threads");
DEFINE_int32(exchange_amount, 1, "Number of proposals to exchange");
DEFINE_int32(exchange_interval, 4, "Number of iterations per exchange");
DEFINE_int32(timeout, 600, "Timeout in seconds");
DEFINE_double(resize_factor, -2,
              "Resize factor, images will be resized by 2^(resize_factor)");

double calcAlpha(const Mat &image, const int pixel, const int foreground_pixel,
                 const int background_pixel) {
  const int IMAGE_WIDTH = image.cols;
  const int IMAGE_HEIGHT = image.rows;

  Vec3b foreground_color = image.at<Vec3b>(foreground_pixel / IMAGE_WIDTH,
                                           foreground_pixel % IMAGE_WIDTH);
  Vec3b background_color = image.at<Vec3b>(background_pixel / IMAGE_WIDTH,
                                           background_pixel % IMAGE_WIDTH);
  Vec3b color = image.at<Vec3b>(pixel / IMAGE_WIDTH, pixel % IMAGE_WIDTH);
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
  return alpha;
}

Mat drawValuesImage(const vector<double> &values, const int IMAGE_WIDTH,
                    const int IMAGE_HEIGHT) {
  Mat image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1);
  for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++)
    image.at<uchar>(pixel / IMAGE_WIDTH, pixel % IMAGE_WIDTH) =
        min(values[pixel] * 256, 255.0);
  return image;
}

void calcWindowMeansAndVars(const std::vector<std::vector<double>> &values,
                            const std::vector<double> &weights,
                            const int IMAGE_WIDTH, const int IMAGE_HEIGHT,
                            const int WINDOW_SIZE,
                            vector<vector<double>> &means,
                            vector<vector<double>> &vars) {
  const int NUM_CHANNELS = values.size();
  vector<vector<double>> weighted_values(
      NUM_CHANNELS, vector<double>(IMAGE_WIDTH * IMAGE_HEIGHT));
  for (int c = 0; c < NUM_CHANNELS; c++)
    transform(values[c].begin(), values[c].end(), weights.begin(),
              weighted_values[c].begin(),
              [](const double &x, const double &y) { return x * y; });
  vector<vector<double>> sum_masks(NUM_CHANNELS);
  for (int c = 0; c < NUM_CHANNELS; c++)
    sum_masks[c] = calcBoxIntegrationMask(values[c], IMAGE_WIDTH, IMAGE_HEIGHT);

  vector<vector<double>> weighted_values2(
      NUM_CHANNELS * NUM_CHANNELS, vector<double>(IMAGE_WIDTH * IMAGE_HEIGHT));
  for (int c_1 = 0; c_1 < NUM_CHANNELS; c_1++)
    for (int c_2 = 0; c_2 < NUM_CHANNELS; c_2++)
      transform(values[c_1].begin(), values[c_1].end(),
                weighted_values[c_2].begin(),
                weighted_values2[c_1 * NUM_CHANNELS + c_2].begin(),
                [](const double &x, const double &y) { return x * y; });
  vector<vector<double>> sum2_masks(NUM_CHANNELS * NUM_CHANNELS);
  for (int c_1 = 0; c_1 < NUM_CHANNELS; c_1++)
    for (int c_2 = 0; c_2 < NUM_CHANNELS; c_2++)
      sum2_masks[c_1 * NUM_CHANNELS + c_2] =
          calcBoxIntegrationMask(weighted_values2[c_1 * NUM_CHANNELS + c_2],
                                 IMAGE_WIDTH, IMAGE_HEIGHT);

  vector<double> weight_sum_mask =
      calcBoxIntegrationMask(weights, IMAGE_WIDTH, IMAGE_HEIGHT);

  means.assign(NUM_CHANNELS, vector<double>(IMAGE_WIDTH * IMAGE_HEIGHT));
  vars.assign(NUM_CHANNELS * NUM_CHANNELS,
              vector<double>(IMAGE_WIDTH * IMAGE_HEIGHT));
  for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++) {
    int x_1 = pixel % IMAGE_WIDTH - (WINDOW_SIZE - 1) / 2;
    int y_1 = pixel / IMAGE_WIDTH - (WINDOW_SIZE - 1) / 2;
    int x_2 = pixel % IMAGE_WIDTH + (WINDOW_SIZE - 1) / 2;
    int y_2 = pixel / IMAGE_WIDTH + (WINDOW_SIZE - 1) / 2;

    int area = calcBoxIntegration(weight_sum_mask, IMAGE_WIDTH, IMAGE_HEIGHT,
                                  x_1, y_1, x_2, y_2);
    vector<double> mean(NUM_CHANNELS);
    for (int c = 0; c < NUM_CHANNELS; c++)
      mean[c] = calcBoxIntegration(sum_masks[c], IMAGE_WIDTH, IMAGE_HEIGHT, x_1,
                                   y_1, x_2, y_2) /
                area;
    vector<double> var(NUM_CHANNELS * NUM_CHANNELS);
    for (int c_1 = 0; c_1 < NUM_CHANNELS; c_1++)
      for (int c_2 = 0; c_2 < NUM_CHANNELS; c_2++)
        var[c_1 * NUM_CHANNELS + c_2] =
            calcBoxIntegration(sum2_masks[c_1 * NUM_CHANNELS + c_2],
                               IMAGE_WIDTH, IMAGE_HEIGHT, x_1, y_1, x_2, y_2) /
                area -
            mean[c_1] * mean[c_2];
    for (int c = 0; c < NUM_CHANNELS; c++)
      means[c][pixel] = mean[c];
    for (int c_1 = 0; c_1 < NUM_CHANNELS; c_1++)
      for (int c_2 = 0; c_2 < NUM_CHANNELS; c_2++)
        vars[c_1 * NUM_CHANNELS + c_2][pixel] = var[c_1 * NUM_CHANNELS + c_2];
  }
}

Mat calcAlphaImage(const Mat &image, const Mat &trimap) {
  const int IMAGE_WIDTH = image.cols;
  const int IMAGE_HEIGHT = image.rows;
  const int NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;

  vector<bool> foreground_mask_vec(NUM_PIXELS, false);
  vector<bool> background_mask_vec(NUM_PIXELS, false);

  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    int color = trimap.at<uchar>(pixel / IMAGE_WIDTH, pixel % IMAGE_WIDTH);
    if (color > 200)
      foreground_mask_vec[pixel] = true;
    if (color < 100)
      background_mask_vec[pixel] = true;
  }

  ImageMask foreground_mask(foreground_mask_vec, IMAGE_WIDTH, IMAGE_HEIGHT);
  ImageMask background_mask(background_mask_vec, IMAGE_WIDTH, IMAGE_HEIGHT);

  vector<int> foreground_boundary_map;
  vector<double> foreground_distance_map;
  foreground_mask.calcBoundaryDistanceMap(foreground_boundary_map,
                                          foreground_distance_map);
  vector<int> background_boundary_map;
  vector<double> background_distance_map;
  background_mask.calcBoundaryDistanceMap(background_boundary_map,
                                          background_distance_map);

  vector<double> alpha_values(NUM_PIXELS);
  vector<double> alpha_confidences(NUM_PIXELS);
  const double COLOR_DIFF_VAR = 100;
  const double MIN_ALPHA_CONFIDENCE = 0.1;
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    if (foreground_mask.at(pixel)) {
      alpha_values[pixel] = 1;
      alpha_confidences[pixel] = 1;
    } else if (background_mask.at(pixel)) {
      alpha_values[pixel] = 0;
      alpha_confidences[pixel] = 1;
    } else {
      int foreground_pixel = foreground_boundary_map[pixel];
      int background_pixel = background_boundary_map[pixel];
      double alpha =
          calcAlpha(image, pixel, foreground_pixel, background_pixel);
      Vec3b color = image.at<Vec3b>(pixel / IMAGE_WIDTH, pixel % IMAGE_WIDTH);
      Vec3b foreground_color = image.at<Vec3b>(foreground_pixel / IMAGE_WIDTH,
                                               foreground_pixel % IMAGE_WIDTH);
      Vec3b background_color = image.at<Vec3b>(background_pixel / IMAGE_WIDTH,
                                               background_pixel % IMAGE_WIDTH);
      double color_diff = 0;
      for (int c = 0; c < 3; c++)
        color_diff += pow(color[c] - (alpha * foreground_color[c] +
                                      (1 - alpha) * background_color[c]),
                          2);
      alpha_values[pixel] = alpha;
      alpha_confidences[pixel] =
          max(exp(-color_diff / (2 * COLOR_DIFF_VAR)), MIN_ALPHA_CONFIDENCE);
      // alpha_confidences[pixel] = 1;
    }
  }

  imwrite("Test/alpha_image_0.bmp",
          drawValuesImage(alpha_values, IMAGE_WIDTH, IMAGE_HEIGHT));
  imwrite("Test/confidence_image_0.bmp",
          drawValuesImage(alpha_confidences, IMAGE_WIDTH, IMAGE_HEIGHT));

  vector<int> window_radiuses;
  vector<double> window_epsilons;
  vector<double> window_weights;
  for (int radius = 3; radius < IMAGE_WIDTH / 2; radius *= 2) {
    window_radiuses.push_back(radius);
    window_epsilons.push_back(0.00001);
    window_weights.push_back(1.0);
    break;
  }

  vector<vector<double>> image_values(3, vector<double>(NUM_PIXELS));
  for (int y = 0; y < IMAGE_HEIGHT; y++) {
    for (int x = 0; x < IMAGE_WIDTH; x++) {
      int pixel = y * IMAGE_WIDTH + x;
      Vec3b image_color = image.at<Vec3b>(y, x);
      for (int c = 0; c < 3; c++)
        image_values[c][pixel] = 1.0 * image_color[c] / 256;
    }
  }

  const int NUM_ITERATIONS = 10;
  const double ALPHA_VAR_VAR = 0.01;
  for (int iteration = 1; iteration <= NUM_ITERATIONS; iteration++) {
    cout << iteration << endl;
    vector<double> alpha_value_sums(IMAGE_WIDTH * IMAGE_HEIGHT, 0);
    vector<double> alpha_value_sums2(IMAGE_WIDTH * IMAGE_HEIGHT, 0);
    vector<double> alpha_confidence_sums(IMAGE_WIDTH * IMAGE_HEIGHT, 0);
    vector<double> alpha_confidence_sums2(IMAGE_WIDTH * IMAGE_HEIGHT, 0);
    Mat alpha_confidence_image =
        drawValuesImage(alpha_confidences, IMAGE_WIDTH, IMAGE_HEIGHT);
    for (int window_index = 0; window_index < window_radiuses.size();
         window_index++) {
      const int radius = window_radiuses[window_index];
      const double epsilon = window_epsilons[window_index];

      vector<double> window_alpha_confidences(NUM_PIXELS);
      Mat filtered_alpha_confidence_image;
      guidedFilter(image, alpha_confidence_image,
                   filtered_alpha_confidence_image, radius, epsilon);
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++)
        window_alpha_confidences[pixel] =
            1.0 *
            filtered_alpha_confidence_image.at<uchar>(pixel / IMAGE_WIDTH,
                                                      pixel % IMAGE_WIDTH) /
            256;
      // imwrite("Test/confidence_image_" + to_string(iteration) + "_" +
      // to_string(radius) + ".bmp", drawValuesImage(window_alpha_confidences,
      // IMAGE_WIDTH, IMAGE_HEIGHT));

      vector<vector<double>> image_alpha_values(3, vector<double>(NUM_PIXELS));
      for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
          int pixel = y * IMAGE_WIDTH + x;
          Vec3b image_color = image.at<Vec3b>(y, x);
          double alpha = alpha_values[pixel];
          double confidence = alpha_confidences[pixel];
          for (int c = 0; c < 3; c++) {
            image_alpha_values[c][pixel] =
                (1.0 * image_color[c] / 256) * alpha * confidence;
          }
        }
      }

      // vector<vector<double> > image_means;
      // vector<vector<double> > image_vars;
      // calcWindowMeansAndVars(image_values, alpha_confidences, IMAGE_WIDTH,
      // IMAGE_HEIGHT, radius * 2 + 1, image_means, image_vars);
      // calcWindowMeansAndVars(image_values, IMAGE_WIDTH, IMAGE_HEIGHT, radius
      // * 2 + 1, image_means, image_vars);
      vector<double> dummy_vars;

      vector<double> alpha_confidence_means;
      calcWindowMeansAndVars(alpha_confidences, IMAGE_WIDTH, IMAGE_HEIGHT,
                             radius * 2 + 1, alpha_confidence_means,
                             dummy_vars);

      vector<vector<double>> image_means(3);
      for (int c = 0; c < 3; c++) {
        vector<double> weighted_image_values(NUM_PIXELS);
        transform(image_values[c].begin(), image_values[c].end(),
                  alpha_confidences.begin(), weighted_image_values.begin(),
                  [](const double &x, const double &y) { return x * y; });
        calcWindowMeansAndVars(weighted_image_values, IMAGE_WIDTH, IMAGE_HEIGHT,
                               radius * 2 + 1, image_means[c], dummy_vars);
        for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++)
          image_means[c][pixel] /= alpha_confidence_means[pixel];
      }

      vector<vector<double>> image_vars(9);
      for (int c_1 = 0; c_1 < 3; c_1++) {
        for (int c_2 = 0; c_2 < 3; c_2++) {
          vector<double> weighted_image_values2(NUM_PIXELS);
          transform(image_values[c_1].begin(), image_values[c_1].end(),
                    image_values[c_2].begin(), weighted_image_values2.begin(),
                    [](const double &x, const double &y) { return x * y; });
          transform(weighted_image_values2.begin(),
                    weighted_image_values2.end(), alpha_confidences.begin(),
                    weighted_image_values2.begin(),
                    [](const double &x, const double &y) { return x * y; });
          calcWindowMeansAndVars(weighted_image_values2, IMAGE_WIDTH,
                                 IMAGE_HEIGHT, radius * 2 + 1,
                                 image_vars[c_1 * 3 + c_2], dummy_vars);
          for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++)
            image_vars[c_1 * 3 + c_2][pixel] =
                image_vars[c_1 * 3 + c_2][pixel] /
                    alpha_confidence_means[pixel] -
                image_means[c_1][pixel] * image_means[c_2][pixel];
        }
      }

      vector<double> alpha_means;
      vector<double> weighted_alpha_values(NUM_PIXELS);
      transform(alpha_values.begin(), alpha_values.end(),
                alpha_confidences.begin(), weighted_alpha_values.begin(),
                [](const double &x, const double &y) { return x * y; });
      calcWindowMeansAndVars(weighted_alpha_values, IMAGE_WIDTH, IMAGE_HEIGHT,
                             radius * 2 + 1, alpha_means, dummy_vars);

      vector<vector<double>> image_alpha_means(3);
      for (int c = 0; c < 3; c++)
        calcWindowMeansAndVars(image_alpha_values[c], IMAGE_WIDTH, IMAGE_HEIGHT,
                               radius * 2 + 1, image_alpha_means[c],
                               dummy_vars);

      for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++) {
        alpha_means[pixel] /= alpha_confidence_means[pixel];
        for (int c = 0; c < 3; c++)
          image_alpha_means[c][pixel] /= alpha_confidence_means[pixel];
      }

      vector<vector<double>> a_values(3, vector<double>(NUM_PIXELS, 0));
      vector<double> b_values(NUM_PIXELS, 0);
      for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++) {
        vector<double> image_alpha_covariance(3);
        vector<vector<double>> image_var(3, vector<double>(3));
        for (int c = 0; c < 3; c++)
          image_alpha_covariance[c] =
              image_alpha_means[c][pixel] -
              image_means[c][pixel] * alpha_means[pixel];
        for (int c_1 = 0; c_1 < 3; c_1++)
          for (int c_2 = 0; c_2 < 3; c_2++)
            image_var[c_1][c_2] =
                image_vars[c_1 * 3 + c_2][pixel] + epsilon * (c_1 == c_2);

        vector<vector<double>> image_var_inverse = calcInverse(image_var);
        vector<double> a_value(3, 0);
        for (int c_1 = 0; c_1 < 3; c_1++)
          for (int c_2 = 0; c_2 < 3; c_2++)
            a_value[c_1] +=
                image_var_inverse[c_1][c_2] * image_alpha_covariance[c_2];
        for (int c = 0; c < 3; c++)
          a_values[c][pixel] = a_value[c];

        // for (int c = 0; c < 3; c++)
        //   if (isnan(float(a_values[c][pixel]))) {
        //     cout << pixel << endl;
        //     exit(1);
        //   }

        double b = alpha_means[pixel];
        for (int c = 0; c < 3; c++)
          b -= a_value[c] * image_means[c][pixel];
        b_values[pixel] = b;
      }

      // double min_a = 1000000;
      // double max_a = -1000000;
      // for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++) {
      //  for (int c = 0; c < 3; c++) {
      //    if (a_values[c][pixel] < min_a)
      //      min_a = a_values[c][pixel];
      //    if (a_values[c][pixel] > max_a)
      //      max_a = a_values[c][pixel];
      //  }
      // }
      // cout << min_a << '\t' << max_a << endl;
      // exit(1);

      vector<vector<double>> a_means(3);
      //      vector<vector<double> > a_vars;
      for (int c = 0; c < 3; c++) {
        vector<double> weighted_a_values = a_values[c];
        // transform(a_values[c].begin(), a_values[c].end(),
        // window_alpha_confidences.begin(), weighted_a_values.begin(), [](const
        // double &x, const double &y) { return x * y; });
        calcWindowMeansAndVars(weighted_a_values, IMAGE_WIDTH, IMAGE_HEIGHT,
                               radius * 2 + 1, a_means[c], dummy_vars);
      }

      vector<double> b_means;
      // vector<double> b_vars;
      vector<double> weighted_b_values = b_values;
      // transform(b_values.begin(), b_values.end(),
      // alpha_confidence_means.begin(), weighted_b_values.begin(), [](const
      // double &x, const double &y) { return x * y; });
      calcWindowMeansAndVars(weighted_b_values, IMAGE_WIDTH, IMAGE_HEIGHT,
                             radius * 2 + 1, b_means, dummy_vars);
      //      calcWindowMeansAndVars(b_values, IMAGE_WIDTH, IMAGE_HEIGHT, radius
      //      * 2 + 1, b_means, b_vars);

      // vector<vector<double> > a_b_means(3);
      // for (int c = 0; c < 3; c++) {
      //   vector<double> a_b_values(NUM_PIXELS);
      //  transform(a_values[c].begin(), a_values[c].end(),
      // b_values.begin(), a_b_values.begin(), [](const double &x, const double
      // &y) { return x * y; });
      //   calcWindowMeansAndVars(a_b_values, IMAGE_WIDTH, IMAGE_HEIGHT, radius
      //   * 2 + 1, a_b_means[c], dummy_vars);
      // }

      {
        Mat alpha_image = Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1);
        for (int y = 0; y < IMAGE_HEIGHT; y++) {
          for (int x = 0; x < IMAGE_WIDTH; x++) {
            int pixel = y * IMAGE_WIDTH + x;
            double alpha = b_means[pixel];
            for (int c = 0; c < 3; c++)
              alpha += a_means[c][pixel] * image_values[c][pixel];
            alpha_image.at<uchar>(y, x) = max(min(alpha * 256, 255.0), 0.0);

            if (iteration == 2 && pixel == 12 * IMAGE_WIDTH + 397 && false) {
              cout << alpha_confidences[pixel] << endl;
              cout << alpha_confidence_means[pixel] << '\t'
                   << alpha_means[pixel] << endl;
              for (int c = 0; c < 3; c++)
                cout << image_alpha_means[c][pixel] << endl;
              for (int c = 0; c < 3; c++)
                cout << image_means[c][pixel] << endl;
              for (int c_1 = 0; c_1 < 3; c_1++)
                for (int c_2 = 0; c_2 < 3; c_2++)
                  cout << image_vars[c_1 * 3 + c_2][pixel] << endl;
              for (int c = 0; c < 3; c++)
                cout << a_means[c][pixel] << endl;
              cout << b_means[pixel] << endl;
              cout << alpha << endl;
              exit(1);
            }
          }
        }

        imwrite("Test/alpha_image_{}_{}.bmp"_format(iteration, radius),
                alpha_image);
      }

      int window_weight = window_weights[window_index];
      for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++) {
        double alpha = b_means[pixel];
        for (int c = 0; c < 3; c++)
          alpha += a_means[c][pixel] * image_values[c][pixel];
        alpha = max(min(alpha, 1.0), 0.0);
        alpha_value_sums[pixel] +=
            alpha * window_weight * window_alpha_confidences[pixel];
        // if (pixel == 12 * IMAGE_WIDTH + 346)
        // cout << pixel << '\t' << alpha << '\t' << alpha_value_sums[pixel] <<
        // endl;

        // double alpha_var = b_vars[pixel];
        // for (int c_1 = 0; c_1 < 3; c_1++)
        //   for (int c_2 = 0; c_2 < 3; c_2++)
        //     alpha_var += image_values[c_1][pixel] * a_vars[c_1 * 3 +
        //     c_2][pixel] * image_values[c_2][pixel];
        // for (int c = 0; c < 3; c++)
        //   alpha_var += 2 * a_b_means[c][pixel];
        // alpha_value_sums2[pixel] += (alpha_var + pow(alpha, 2)) *
        // window_weight * window_alpha_confidences[pixel];

        alpha_confidence_sums[pixel] +=
            window_weight * window_alpha_confidences[pixel];
        alpha_confidence_sums2[pixel] += window_weight *
                                         window_alpha_confidences[pixel] *
                                         window_alpha_confidences[pixel];
      }

      // for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
      //  for (int c = 0; c < 3; c++)
      //    pixel_window_a_means[pixel][window_index][c] =
      // a_means[c][pixel];
      //  for (int c_1 = 0; c_1 < 3; c_1++)
      //     for (int c_2 = 0; c_2 < 3; c_2++)
      //      pixel_window_a_vars[pixel][window_index][c_1 * 3 + c_2] =
      // a_vars[c_1 * 3 + c_2][pixel];
      //  pixel_window_b_means[pixel][window_index] = b_means[pixel];
      //  pixel_window_b_vars[pixel][window_index] = b_vars[pixel];
      //  for (int c = 0; c < 3; c++)
      //    pixel_window_a_b_means[pixel][window_index][c] =
      // a_b_means[c][pixel];
      // }
    }

    for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
      double alpha_mean =
          alpha_confidence_sums[pixel] != 0
              ? alpha_value_sums[pixel] / alpha_confidence_sums[pixel]
              : rand() / numeric_limits<int>::max();
      alpha_mean = max(min(alpha_mean, 1.0), 0.0);
      alpha_values[pixel] = alpha_mean;
      alpha_confidences[pixel] = alpha_confidence_sums[pixel] != 0
                                     ? max(alpha_confidence_sums2[pixel] /
                                               alpha_confidence_sums[pixel],
                                           MIN_ALPHA_CONFIDENCE)
                                     : 0;
      // if (pixel == 12 * IMAGE_WIDTH + 346)
      //   cout << pixel << '\t' << alpha_mean << '\t' <<
      //   alpha_value_sums[pixel] << endl;

      // double alpha_var = alpha_value_sums2[pixel] /
      // alpha_confidence_sums[pixel] - pow(alpha_mean, 2);

      // alpha_confidences[pixel] = max(exp(-alpha_var / (2 * ALPHA_VAR_VAR)),
      // MIN_ALPHA_CONFIDENCE);
      // alpha_confidences[pixel] = 1;
    }

    // for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    //   vector<vector<double> > window_a_means = pixel_window_a_means[pixel];
    //   vector<vector<double> > window_a_vars = pixel_window_a_vars[pixel];
    //   vector<double> window_b_means = pixel_window_b_means[pixel];
    //   vector<double> window_b_vars = pixel_window_b_vars[pixel];
    //   vector<vector<double> > window_a_b_means =
    //   pixel_window_a_b_means[pixel];

    //   vector<double> a_mean_sum(3, 0);
    //   vector<double> a_var_sum(9, 0);
    //   double b_mean_sum = 0;
    //   double b_var_sum = 0;
    //   vector<double> a_b_mean_sum(3, 0);
    //   int area_sum = 0;
    //   for (int window_index = 0; window_index < window_radiuses.size();
    //   window_index++) {
    //  const int radius = window_radiuses[window_index];
    //  int area = pow(radius * 2 + 1, 2);
    //  for (int c = 0; c < 3; c++)
    //    a_mean_sum[c] += window_a_means[window_index][c] * area;
    //     for (int c_1 = 0; c_1 < 3; c_1++)
    //    for (int c_2 = 0; c_2 < 3; c_2++)
    //      a_var_sum[c_1 * 3 + c_2] += window_a_vars[window_index][c_1 * 3 +
    // c_2] * area;
    //  b_mean_sum += window_b_means[window_index] * area;
    //  b_var_sum += window_b_vars[window_index] * area;
    //  for (int c = 0; c < 3; c++)
    //       a_b_mean_sum[c] += window_a_b_means[window_index][c] * area;
    //  area_sum += area;
    //   }

    //   vector<double> a_mean(3, 0);
    //   for (int c = 0; c < 3; c++)
    //     a_mean[c] = a_mean_sum[c] / area_sum;
    //   vector<double> a_var(9, 0);
    //   for (int c_1 = 0; c_1 < 3; c_1++)
    //     for (int c_2 = 0; c_2 < 3; c_2++)
    //       a_var[c_1 * 3 + c_2] = a_var_sum[c_1 * 3 + c_2] / area_sum;
    //   double b_mean = b_mean_sum / area_sum;
    //   double b_var = b_var_sum / area_sum;
    //   vector<double> a_b_mean(3, 0);
    //   for (int c = 0; c < 3; c++)
    //     a_b_mean[c] = a_b_mean_sum[c] / area_sum;

    //   double alpha = b_mean;
    //   for (int c = 0; c < 3; c++)
    //  alpha += a_mean[c] * image_values[c][pixel];
    //   alpha_values[pixel] = alpha;
    //   double alpha_var = b_var;
    //   for (int c_1 = 0; c_1 < 3; c_1++)
    //     for (int c_2 = 0; c_2 < 3; c_2++)
    //    alpha_var += image_values[c_1][pixel] * a_var[c_1 * 3 + c_2] *
    // image_values[c_2][pixel];
    //   for (int c = 0; c < 3; c++)
    //  alpha_var += 2 * a_b_mean[c];
    //   alpha_confidences[pixel] = max(exp(-alpha_var / (2 * ALPHA_VAR_VAR)),
    //   MIN_ALPHA_CONFIDENCE);
    // }
    for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
      if (foreground_mask.at(pixel)) {
        alpha_values[pixel] = 1;
        alpha_confidences[pixel] = 1;
      } else if (background_mask.at(pixel)) {
        alpha_values[pixel] = 0;
        alpha_confidences[pixel] = 1;
      }
    }
    imwrite("Test/alpha_image_{}.bmp"_format(iteration),
            drawValuesImage(alpha_values, IMAGE_WIDTH, IMAGE_HEIGHT));
    imwrite("Test/confidence_image_{}.bmp"_format(iteration),
            drawValuesImage(alpha_confidences, IMAGE_WIDTH, IMAGE_HEIGHT));
  }

  Mat alpha_image = Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1);
  for (int y = 0; y < IMAGE_HEIGHT; y++) {
    for (int x = 0; x < IMAGE_WIDTH; x++) {
      int pixel = y * IMAGE_WIDTH + x;
      alpha_image.at<uchar>(y, x) =
          max(min(alpha_values[pixel] * 256, 255.0), 0.0);
    }
  }
  return alpha_image;
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  auto console = spdlog::stdout_color_mt("console");
#if 0
  Mat image = imread(FLAGS_base_dir + "/Training/Images/GT24.png");
  CHECK_NOTNULL(image.data);
  Mat alpha_image = imread(FLAGS_base_dir + "/Test/alpha_image_3.bmp", 0);
  Mat trimap = imread(FLAGS_base_dir + "/Training/Trimap1/GT24.png", 0);
  CHECK_NOTNULL(trimap.data);

#if 0
      const int IMAGE_WIDTH = 100;
      const int IMAGE_HEIGHT = 100;
      image = Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
      trimap = Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1);
      for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
          if (abs(x - IMAGE_WIDTH / 2) < 20 && abs(y - IMAGE_HEIGHT / 2) < 20) {
            image.at<Vec3b>(y, x) = Vec3b(255, 255, 255);
            trimap.at<uchar>(y, x) = 255;
          } else if (abs(x - IMAGE_WIDTH / 2) > 30 ||
                     abs(y - IMAGE_HEIGHT / 2) > 30) {
            image.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
            trimap.at<uchar>(y, x) = 0;
          } else {
            if (x == IMAGE_WIDTH / 2)
              image.at<Vec3b>(y, x) = Vec3b(255, 255, 255);
            else if (y == IMAGE_HEIGHT / 2)
              image.at<Vec3b>(y, x) = Vec3b(128, 128, 128);
            else
              image.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
            trimap.at<uchar>(y, x) = 128;
          }
        }
      }
      imwrite("Test/image.bmp", image);
      imwrite("Test/trimap.bmp", trimap);
#endif

  alpha_image = imread(FLAGS_base_dir + "/Test/alpha_image_0.bmp", 0);

  // resize(image, image, Size(image.cols / 3, image.rows / 3));
  // resize(alpha_image, alpha_image, Size(alpha_image.cols / 3,
  // alpha_image.rows / 3));
  // resize(trimap, trimap, Size(trimap.cols / 3, trimap.rows / 3));

  Mat filtered_alpha_image = calcAlphaImage(image, trimap);
  // Mat filtered_alpha_image;
  // guidedFilter(image, alpha_image, filtered_alpha_image, 3, 0.0001);
  imwrite(FLAGS_base_dir + "/Test/filtered_alpha_image.bmp", filtered_alpha_image);
  exit(1);
#endif
  double training_error = 0;
  for (int trimap_index = 1; trimap_index <= 1; trimap_index++) {
    string input_directory = FLAGS_base_dir + "/Input/";
    string output_directory = FLAGS_base_dir + "/Output/";
    string image_directory = "Images/";
    string trimap_directory = "Trimap{}/"_format(trimap_index);
    cv::Mat image, trimap;
    std::string output_alpha_image_filename, output_guided_alpha_image_filename;
#if 0
      LOG(INFO) << trimap_index << '\t' << *image_name_it << endl;
      output_alpha_image_filename << output_directory << trimap_directory
                                  << *image_name_it << ".png";
      LOG(INFO) << "Ouput dir: " << output_alpha_image_filename.str()
                << std::endl;
      if (!imread(output_alpha_image_filename.str()).empty())
        continue;
      LOG(INFO) << "Input dir: "
                << (input_directory + image_directory + *image_name_it + ".png")
                << endl;
      image =
          imread(input_directory + image_directory + *image_name_it + ".png");
      CHECK_NOTNULL(image.data);
      trimap = imread(
          input_directory + trimap_directory + *image_name_it + ".png", 0);
      CHECK_NOTNULL(image.data);
      // cout << 7270 / image.cols << '\t' << 7270 % image.cols << endl;
#else
    const auto &training_ID = FLAGS_img_name;
    image =
        imread("{}/Training/Images/{}.png"_format(FLAGS_base_dir, training_ID));
    CHECK_NOTNULL(image.data);

    const std::string trimap_file_name = "{}/Training/Trimap{}/{}.png"_format(
        FLAGS_base_dir, trimap_index, training_ID);
    trimap = imread(trimap_file_name, 0);
    CHECK_NOTNULL(trimap.data);
    Mat alpha_ground_truth = imread(
        "{}/Training/GroundTruth/{}.png"_format(FLAGS_base_dir, training_ID),
        0);
    CHECK_NOTNULL(alpha_ground_truth.data);
    string image_identifier = training_ID + "_1";
    LOG(INFO) << "Running training round on " << image_identifier;
    output_alpha_image_filename =
        "{}/Training/Output/{}.png"_format(FLAGS_base_dir, image_identifier);

    LOG(INFO) << "Ouput Name: " << output_alpha_image_filename;

    output_guided_alpha_image_filename =
        "{}/Training/Output/{}_guided.png"_format(FLAGS_base_dir,
                                                  image_identifier);

    LOG(INFO) << "Guided Ouput Name: " << output_guided_alpha_image_filename;
#endif

#if 1
    const double resize_factor = pow(2, FLAGS_resize_factor);
    resize(image, image, Size(), resize_factor, resize_factor);
    resize(trimap, trimap, Size(), resize_factor, resize_factor);
    if (alpha_ground_truth.data)
      resize(alpha_ground_truth, alpha_ground_truth, Size(), resize_factor,
             resize_factor);

    LOG(INFO) << "Resized image: " << image.size();
    LOG(INFO) << "Resized trimap: " << trimap.size();
    LOG_IF(INFO, alpha_ground_truth.data) << "Resized alpha_ground_truth: "
                                          << alpha_ground_truth.size();
#endif

    imwrite(FLAGS_base_dir + "/Test/image.bmp", image);
    imwrite(FLAGS_base_dir + "/Test/trimap.bmp", trimap);

    vector<bool> foreground_mask_vec(image.cols * image.rows, false);
    vector<bool> background_mask_vec(image.cols * image.rows, false);
    for (int pixel = 0; pixel < image.cols * image.rows; pixel++) {
      int color = trimap.at<uchar>(pixel / image.cols, pixel % image.cols);
      if (color > 200)
        foreground_mask_vec[pixel] = true;
      if (color < 100)
        background_mask_vec[pixel] = true;
    }

    ImageMask foreground_mask(foreground_mask_vec, image.cols, image.rows);
    ImageMask background_mask(background_mask_vec, image.cols, image.rows);

    ParallelFusion::ParallelFusionOption pipeline_option;
    pipeline_option.max_iteration = 250;
    pipeline_option.num_threads = FLAGS_num_threads;
    pipeline_option.timeout = std::chrono::seconds(FLAGS_timeout);
    typedef AlphaMattingLabelSpace<long> Space;
    typedef ParallelFusion::ParallelFusionPipeline<Space> Pipeline;

    auto cost_functor = AlphaMattingCostFunctor::Create(
        image, foreground_mask, background_mask, image_identifier);

    auto solver = FusionSpaceSolver::Create(
        image.cols * image.rows, cost_functor->getPixelNeighbors(),
        cost_functor, pipeline_option.timeout, 200);
    Pipeline::SolverSet solvers(
        static_cast<size_t>(pipeline_option.num_threads));
    Pipeline::GeneratorSet generators(
        static_cast<size_t>(pipeline_option.num_threads));
    vector<ParallelFusion::ThreadOption> threadOptions(
        static_cast<size_t>(pipeline_option.num_threads));
    vector<Space> initials(static_cast<size_t>(pipeline_option.num_threads));

    std::random_device rng;
    for (int i = 0; i < pipeline_option.num_threads; ++i) {
      initials[i].init(image.cols * image.rows);
      solvers[i] = solver;

      auto tmp = AlphaMattingProposalGenerator::Create(image, foreground_mask,
                                                       background_mask, rng());
      tmp->setNeighbors(cost_functor->getPixelNeighbors());
      generators[i] = tmp;

      threadOptions[i].kTotal = FLAGS_num_proposals;
      threadOptions[i].kOtherThread = FLAGS_exchange_amount;
      threadOptions[i].solution_exchange_interval = FLAGS_exchange_interval;
    }

    // FusionSpaceSolver solver(image.cols * image.rows,
    //                          cost_functor.getPixelNeighbors(),
    //                          cost_functor,
    //                          proposal_generator, 200);
    // FusionSpaceSolver solver(image.cols * image.rows,
    // findNeighborsForAllPixels(image.cols, image.rows), cost_functor,
    // proposal_generator, 200);

    vector<double> foreground_distance_map;
    vector<int> foreground_boundary_map;
    foreground_mask.calcBoundaryDistanceMap(foreground_boundary_map,
                                            foreground_distance_map);

    vector<double> background_distance_map;
    vector<int> background_boundary_map;
    background_mask.calcBoundaryDistanceMap(background_boundary_map,
                                            background_distance_map);

    vector<long> initial_solution(image.cols * image.rows);
    for (int pixel = 0; pixel < image.cols * image.rows; pixel++) {
      if (foreground_mask.at(pixel) || background_mask.at(pixel))
        initial_solution[pixel] =
            static_cast<long>(pixel) * (image.cols * image.rows) + pixel;
      else
        initial_solution[pixel] =
            static_cast<long>(foreground_boundary_map[pixel]) *
                (image.cols * image.rows) +
            background_boundary_map[pixel];
    }

    for (auto &init : initials)
      init.setSingleLabels(initial_solution);

    ParallelFusion::ParallelFusionPipeline<Space> pipline(pipeline_option);
    pipline.runParallelFusion(initials, generators, solvers, threadOptions);
    ParallelFusion::SolutionType<Space> solution;
    pipline.getBestLabeling(solution);
    LOG(INFO) << "Final solution energy: {:.4f}"_format(solution.first);

    auto &thread_obs = pipline.getAllThreadProfiles();
    std::vector<std::tuple<double, double, int>> all_obs;

    double previous_energy = 1e100;
    int tid = 0;
    for (auto &t_profile : thread_obs) {
      for (auto &obs : t_profile) {
        all_obs.emplace_back(obs.first, obs.second, tid);
      }
      ++tid;
    }

    std::sort(all_obs.begin(), all_obs.end(),
              [](std::tuple<double, double, int> &a,
                 std::tuple<double, double, int> &b) {
                return std::get<0>(a) < std::get<0>(b);
              });

    std::ofstream raw("raw-time-v-energy.csv", std::ios::out);
    for (auto &obs : all_obs)
      fmt::print(raw, "{}, {}, {}\n", std::get<0>(obs), std::get<1>(obs),
                 std::get<2>(obs));
    raw.close();

#if 1
    const auto &current_solution = solution.second;
    Mat alpha_image(image.rows, image.cols, CV_8UC1);
    double error = 0;
    int num_unknown_pixels = 0;
    for (int pixel = 0; pixel < image.cols * image.rows; pixel++) {
      alpha_image.at<uchar>(pixel / image.cols, pixel % image.cols) =
          cost_functor->calcAlpha(pixel, current_solution(pixel, 0)) * 255;
      if (foreground_mask.at(pixel) == false &&
          background_mask.at(pixel) == false && alpha_ground_truth.data) {
        error +=
            pow(cost_functor->calcAlpha(pixel, current_solution(pixel, 0)) -
                    1.0 *
                        alpha_ground_truth.at<uchar>(pixel / image.cols,
                                                     pixel % image.cols) /
                        255,
                2);
        num_unknown_pixels++;
      }
    }

    LOG_IF(INFO, alpha_ground_truth.data)
        << "Average error: " << sqrt(error / num_unknown_pixels) << endl;

    LOG(INFO) << "Ouput Name: " << output_alpha_image_filename;
    imwrite(output_alpha_image_filename, alpha_image);

    auto guided_filter = cv::ximgproc::createGuidedFilter(image, 8, 0.1 * 0.1);
    cv::Mat guided_alpha_image;
    guided_filter->filter(alpha_image, guided_alpha_image);

    error = 0;
    num_unknown_pixels = 0;
    for (int pixel = 0;
         pixel < image.cols * image.rows && alpha_ground_truth.data; pixel++) {
      if (foreground_mask.at(pixel) == false &&
          background_mask.at(pixel) == false && alpha_ground_truth.data) {
        error += pow(guided_alpha_image.at<uchar>(pixel / image.cols,
                                                  pixel % image.cols) /
                             255.0 -

                         alpha_ground_truth.at<uchar>(pixel / image.cols,
                                                      pixel % image.cols) /
                             255.0,
                     2);
        num_unknown_pixels++;
      }
    }

    LOG_IF(INFO, alpha_ground_truth.data)
        << "Average guided error: " << sqrt(error / num_unknown_pixels) << endl;
    training_error += sqrt(error / num_unknown_pixels);

    LOG(INFO) << "Guided Ouput Name: " << output_guided_alpha_image_filename;
    imwrite(output_guided_alpha_image_filename, guided_alpha_image);

/*stringstream solution_filename;
solution_filename << "Cache/solution_" << iteration << ".txt";
ofstream solution_out_str(solution_filename.str());
for (vector<long>::const_iterator pixel_it = current_solution.begin();
     pixel_it != current_solution.end(); pixel_it++)
  if (pixel_it - current_solution.begin() !=
      *pixel_it / (image.cols * image.rows))
    solution_out_str
        << (pixel_it - current_solution.begin()) % image.cols << '\t'
        << (pixel_it - current_solution.begin()) / image.cols << '\t'
        << *pixel_it / (image.cols * image.rows) % image.cols << '\t'
        << *pixel_it / (image.cols * image.rows) / image.cols << '\t'
        << *pixel_it % (image.cols * image.rows) % image.cols << '\t'
        << *pixel_it % (image.cols * image.rows) / image.cols << '\t'
        << cost_functor.calcAlpha(pixel_it - current_solution.begin(),
                                  *pixel_it)
        << endl;
}*/

// Mat output_alpha_image(image.rows, image.cols, CV_8UC1);
// double error = 0;
// int num_unknown_pixels = 0;
// for (int pixel = 0; pixel < image.cols * image.rows; pixel++) {
//   //cout << pixel << '\t' << pixel << '\t' << pixel << '\t' << pixel
//   <<
//   endl;
//   output_alpha_image.at<uchar>(pixel / image.cols, pixel %
//   image.cols)
//   = cost_functor.calcAlpha(pixel, current_solution[pixel]) * 255;
//   if (foreground_mask.at(pixel) == false && background_mask.at(pixel)
//   == false) {
//  error += pow(cost_functor.calcAlpha(pixel,
// current_solution[pixel]) - 1.0 * alpha_ground_truth.at<uchar>(pixel /
// image.cols, pixel % image.cols) / 255, 2);
//  num_unknown_pixels++;
//   }
// }
// cout << sqrt(error / num_unknown_pixels) << endl;

// stringstream output_alpha_image_filename;
// output_alpha_image_filename << output_directory << trimap_directory
// <<
// *image_name_it << ".png";

#endif
  }
  fmt::print("Final error: {:.5f}", training_error);
  return 0;
}
