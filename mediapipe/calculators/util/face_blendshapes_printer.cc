#include "face_blendshapes_printer.h"
#include <iostream>
#include <mediapipe/util/json.hpp>
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

#include <winsock2.h> // For Windows sockets
#include <ws2tcpip.h> // For inet_pton
#include <string>


#pragma comment(lib, "ws2_32.lib") // Link against Winsock library

namespace mediapipe {

const std::vector<int> FaceBlendshapesPrinter::FACE_LANDMARKS = {
    10, 297, 284, 389, 454, 361, 397, 378, 152, 149, 172, 132,
    234, 162, 54, 67, 159, 157, 133, 154, 145, 163, 33, 161,
    386, 388, 263, 390, 374, 381, 362, 384, 12, 271, 291, 403,
    15, 179, 61, 41, 164, 473, 468
};

absl::Status FaceBlendshapesPrinter::GetContract(CalculatorContract* cc) {
  cc->Inputs().Tag("IMAGE_SIZE").Set<std::pair<int, int>>(); 
  cc->Inputs().Tag("POSE_LANDMARKS").Set<NormalizedLandmarkList>();
  cc->Inputs().Tag("LEFT_HAND_LANDMARKS").Set<NormalizedLandmarkList>();
  cc->Inputs().Tag("RIGHT_HAND_LANDMARKS").Set<NormalizedLandmarkList>();
  cc->Inputs().Tag("FACE_LANDMARKS").Set<NormalizedLandmarkList>();
  cc->Inputs().Tag("FACE_BLENDSHAPES").Set<ClassificationList>();
  return absl::OkStatus();
}

absl::Status FaceBlendshapesPrinter::Process(CalculatorContext* cc) {
  nlohmann::json json_data;
  int height = 1080;
  int width = 1920;

  if (!cc->Inputs().Tag("IMAGE_SIZE").IsEmpty()) {
    const auto& image_size = cc->Inputs().Tag("IMAGE_SIZE").Get<std::pair<int, int>>();
    width = image_size.first;
    height = image_size.second;
    json_data["Res"] = { {"x", width}, {"y", height} };
  }

  float ratio = (width > 0) ? static_cast<float>(height) / width : 0; // Avoid division by zero

    if (!cc->Inputs().Tag("POSE_LANDMARKS").IsEmpty()) {
    const auto& landmarks = cc->Inputs().Tag("POSE_LANDMARKS").Get<NormalizedLandmarkList>();
    json_data["Body"] = BodyLandmarksToJson(landmarks,ratio);
  }

  if (!cc->Inputs().Tag("LEFT_HAND_LANDMARKS").IsEmpty()) {
    const auto& left_hand_landmarks = cc->Inputs().Tag("LEFT_HAND_LANDMARKS").Get<NormalizedLandmarkList>();
    json_data["LHand"] = HandLandmarksToJson(left_hand_landmarks,ratio);
  }

  if (!cc->Inputs().Tag("RIGHT_HAND_LANDMARKS").IsEmpty()) {
    const auto& right_hand_landmarks = cc->Inputs().Tag("RIGHT_HAND_LANDMARKS").Get<NormalizedLandmarkList>();
    json_data["RHand"] = HandLandmarksToJson(right_hand_landmarks,ratio);
  }

  if (!cc->Inputs().Tag("FACE_LANDMARKS").IsEmpty()) {
    const auto& face_landmarks = cc->Inputs().Tag("FACE_LANDMARKS").Get<NormalizedLandmarkList>();
    json_data["Face"] = FaceLandmarksToJson(face_landmarks,ratio,FACE_LANDMARKS);
  }

  if (!cc->Inputs().Tag("FACE_BLENDSHAPES").IsEmpty()) {
    const auto& blendshapes = cc->Inputs().Tag("FACE_BLENDSHAPES").Get<ClassificationList>();
    json_data["BlendShape"] = ClassificationListToJson(blendshapes);
  }



    // UDP Sending Section
    SOCKET sockfd;
    sockaddr_in server_addr;
    WSADATA wsaData;

    // Initialize Winsock
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "Failed to initialize Winsock." << std::endl;
        return absl::UnknownError("Winsock initialization failed.");
    }

    // Create socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd == INVALID_SOCKET) {
        std::cerr << "Error opening socket." << std::endl;
        WSACleanup();
        return absl::UnknownError("Socket creation failed.");
    }

    // Set up the server address
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(12500);
    inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);

    // Craft JSON and send it over UDP
    std::string json_string = json_data.dump();
    int send_result = sendto(sockfd, json_string.c_str(), json_string.length(), 0,
                             (struct sockaddr*)&server_addr, sizeof(server_addr));

    if (send_result == SOCKET_ERROR) {
        std::cerr << "Error sending data." << std::endl;
        closesocket(sockfd);
        WSACleanup();
        return absl::UnknownError("Failed to send data.");
    } else {
        std::cout << "Sent " << send_result << " bytes to UDP port 12500." << std::endl;
    }

    // Cleanup
    closesocket(sockfd);
    WSACleanup();
  
  // std::cout << json_data.dump(4) << std::endl;
  return absl::OkStatus();
}

nlohmann::json FaceBlendshapesPrinter::NormalizedLandmarkListToJson(const NormalizedLandmarkList& landmarks) const {
  nlohmann::json json_landmarks = nlohmann::json::array();
  
  for (const auto& landmark : landmarks.landmark()) {
    json_landmarks.push_back({
      {"x", landmark.x()},
      {"y", landmark.y()},
      {"z", landmark.z()},
      {"visibility", landmark.visibility()},
      {"presence", landmark.presence()}
    });
  }
  return json_landmarks;
}

nlohmann::json FaceBlendshapesPrinter::ClassificationListToJson(const ClassificationList& classification_list) const {
  std::vector<float> values;
    // Start from the second element and populate the `values` vector.
  for (size_t i = 1; i < classification_list.classification_size(); ++i) {
    values.push_back(classification_list.classification(i).score());
  }
  
  values.push_back(0.0f);
  return values;
}

// Converts body landmarks to JSON (update to be a member function)
nlohmann::json FaceBlendshapesPrinter::BodyLandmarksToJson(const NormalizedLandmarkList& landmarks, float ratio) const {
    nlohmann::json json_array = nlohmann::json::array();
    for (int i = 0; i < landmarks.landmark_size(); ++i) {
        const auto& lm = landmarks.landmark(i); // Accessing each landmark correctly
        json_array.push_back({
            {"pre", round(lm.presence() * 1000) / 1000}, // Using the getter correctly
            {"vis", round(lm.visibility() * 1000) / 1000}, // Using the getter correctly
            {"pos", {
                {"x", round((lm.x() - 0.5) * 10000) / 10000}, // Using the getter correctly
                {"y", round((-1 * (lm.y() - 0.5) * ratio) * 10000) / 10000}, // Using the getter correctly
                {"z", round(-lm.z() * 10000) / 10000} // Using the getter correctly
            }}
        });
    }
    return json_array;
}

nlohmann::json FaceBlendshapesPrinter::FaceLandmarksToJson(const NormalizedLandmarkList& landmarks, float ratio, const std::vector<int>& face_landmarks_indices) const{
    nlohmann::json json_array = nlohmann::json::array();
    
    for (int i : face_landmarks_indices) { // Iterate over the provided indices for face landmarks
        const auto& landmark = landmarks.landmark(i); // Access using index
        json_array.push_back({
            {"id", i},
            {"pos", {
                {"x", round((landmark.x() - 0.5) * 10000) / 10000}, // Adjusted rounding
                {"y", round((-1 * (landmark.y() - 0.5) * ratio) * 10000) / 10000}, // Adjusted rounding
                {"z", round(-landmark.z() * 10000) / 10000} // Adjusted rounding
            }}
        });
    }
    
    return json_array;
}

nlohmann::json FaceBlendshapesPrinter::HandLandmarksToJson(const NormalizedLandmarkList& landmarks, float ratio) const {
    nlohmann::json json_array = nlohmann::json::array();
    for (int i = 0; i < landmarks.landmark_size(); ++i) {
        const auto& lm = landmarks.landmark(i); // Access each landmark
        json_array.push_back({
            {"x", round((lm.x() - 0.5) * 10000) / 10000}, // Adjusted rounding
            {"y", round((-1 * (lm.y() - 0.5) * ratio) * 10000) / 10000}, // Adjusted rounding
            {"z", round(-lm.z() * 10000) / 10000} // Adjusted rounding
        });
    }
    return json_array;
}

REGISTER_CALCULATOR(FaceBlendshapesPrinter);

}  // namespace mediapipe
