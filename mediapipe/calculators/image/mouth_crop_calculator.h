#ifndef MEDIAPIPE_CALCULATORS_IMAGE_CHEEK_CROP_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_IMAGE_CHEEK_CROP_CALCULATOR_H_

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"  // For NormalizedLandmarkList
#include <opencv2/highgui.hpp>
namespace mediapipe {

class MouthCropCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);
  ::mediapipe::Status Process(CalculatorContext* cc) override;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_IMAGE_CHEEK_CROP_CALCULATOR_H_
