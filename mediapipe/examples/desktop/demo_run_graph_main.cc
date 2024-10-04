// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <cstdlib>
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfobjects.h>
#include <mfplay.h>
#include <mfreadwrite.h>
#include <mferror.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";

// Command Line
ABSL_FLAG(std::string, calculator_graph_config_file, "holistic_tracking_cpu.pbtxt",
          "Name of file containing text format CalculatorGraphConfig proto.");

ABSL_FLAG(int, id, 0, "Camera index to use (default is 0)");
ABSL_FLAG(int, width, 640, "Camera resolution width");
ABSL_FLAG(int, height, 480, "Camera resolution height");
ABSL_FLAG(int, fps, 30, "Camera fps");
ABSL_FLAG(int, port, 12500, "UDP port to Unity");
ABSL_FLAG(std::string, executor, "Invalid", "Name of executor");

void EnumerateVideoCaptureDevices() {
    IMFAttributes* pAttributes = NULL;
    IMFActivate** ppDevices = NULL;
    UINT32 deviceCount = 0;

    // Create an attribute store to specify enumeration parameters.
    HRESULT hr = MFCreateAttributes(&pAttributes, 1);
    if (FAILED(hr)) {
        std::cerr << "Failed to create attributes." << std::endl;
        return;
    }

    // Request video capture devices.
    hr = pAttributes->SetGUID(
        MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
        MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID
    );

    if (FAILED(hr)) {
        std::cerr << "Failed to set attribute source type." << std::endl;
        pAttributes->Release();
        return;
    }

    // Enumerate devices.
    hr = MFEnumDeviceSources(pAttributes, &ppDevices, &deviceCount);

    if (SUCCEEDED(hr)) {
        for (UINT32 i = 0; i < deviceCount; i++) {
            WCHAR* friendlyName = NULL;
            UINT32 nameLength = 0;

            hr = ppDevices[i]->GetAllocatedString(
                MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME,
                &friendlyName, 
                &nameLength
            );

            if (SUCCEEDED(hr)) {
                // Convert WCHAR* to std::wstring and then to std::string
                std::wstring ws(friendlyName);
                std::string deviceName(ws.begin(), ws.end());
                std::cout << "Device " << i << ": " << deviceName << std::endl;
                CoTaskMemFree(friendlyName);
            }
            ppDevices[i]->Release();
        }
        CoTaskMemFree(ppDevices);
    } else {
        std::cerr << "Failed to enumerate devices." << std::endl;
    }
    pAttributes->Release();
}


absl::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      absl::GetFlag(FLAGS_calculator_graph_config_file),
      &calculator_graph_config_contents));
  ABSL_LOG(INFO) << "Get calculator graph config contents: "
                 << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  ABSL_LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  ABSL_LOG(INFO) << "Initialize the camera or load the video.";
  cv::VideoCapture capture;
  capture.open(absl::GetFlag(FLAGS_id),cv::CAP_DSHOW);
  RET_CHECK(capture.isOpened());

  // GlobalConfig::Port = absl::GetFlag(FLAGS_port);

  cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
  capture.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(absl::GetFlag(FLAGS_width)));
  capture.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(absl::GetFlag(FLAGS_height)));
  capture.set(cv::CAP_PROP_FPS, static_cast<double>(absl::GetFlag(FLAGS_fps)));


  ABSL_LOG(INFO) << "Start running the calculator graph.";
  MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                      graph.AddOutputStreamPoller(kOutputStream));
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  ABSL_LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;
  while (grab_frames) {
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) {
      ABSL_LOG(INFO) << "Ignore empty frames from camera.";
      continue;
    }
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    // if (!load_video) {
    //   cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
    // }

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Send image packet into the graph.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        kInputStream, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp_us))));

    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    if (!poller.Next(&packet)) break;
    auto& output_frame = packet.Get<mediapipe::ImageFrame>();

    // Convert back to opencv for display or saving.
    cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    cv::imshow(kWindowName, output_frame_mat);
    // Press any key to exit.
    const int pressed_key = cv::waitKey(5);
    if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
  }

  ABSL_LOG(INFO) << "Shutting down.";
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
  // Initialize logging and parse command line flags.
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);

  std::string executor_value = absl::GetFlag(FLAGS_executor);
  if (executor_value != "AvaKit") {
      std::cout << "The executor value is not 'AvaKit'. It is: " << executor_value << std::endl;
      return EXIT_FAILURE;
  }

  PrintCommandLine();


  // Initialize COM library
  HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
  if (FAILED(hr)) {
    ABSL_LOG(ERROR) << "Failed to initialize COM library.";
    return EXIT_FAILURE;
  }

  // Initialize Media Foundation
  hr = MFStartup(MF_VERSION);
  if (FAILED(hr)) {
    ABSL_LOG(ERROR) << "Failed to initialize Media Foundation.";
    CoUninitialize();
    return EXIT_FAILURE;
  }

  EnumerateVideoCaptureDevices();

  // Run the MediaPipe graph
  absl::Status run_status = RunMPPGraph();
  
  // Uninitialize Media Foundation
  MFShutdown();
  
  // Uninitialize COM library
  CoUninitialize();

  if (!run_status.ok()) {
    ABSL_LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    ABSL_LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
