// #include "mediapipe/calculators/util/global_config.h"
#include "face_blendshapes_printer.h"
#include <iostream>
#include <mediapipe/util/json.hpp>
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "absl/flags/flag.h"

#include <winsock2.h> // For Windows sockets
#include <ws2tcpip.h> // For inet_pton
#include <string>

#pragma comment(lib, "ws2_32.lib") // Link against Winsock library
ABSL_FLAG(int, port, 12500, "UDP port to Unity");

namespace mediapipe {

// extern int getUDPport();

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
    json_data["Body"] = BodyLandmarksToJson(landmarks, ratio);
} else {
    json_data["Body"] = nullptr; // None equivalent in JSON
}

if (!cc->Inputs().Tag("LEFT_HAND_LANDMARKS").IsEmpty()) {
    const auto& left_hand_landmarks = cc->Inputs().Tag("LEFT_HAND_LANDMARKS").Get<NormalizedLandmarkList>();
    json_data["LHand"] = HandLandmarksToJson(left_hand_landmarks, ratio);
} else {
    json_data["LHand"] = nullptr; // None equivalent in JSON
}

if (!cc->Inputs().Tag("RIGHT_HAND_LANDMARKS").IsEmpty()) {
    const auto& right_hand_landmarks = cc->Inputs().Tag("RIGHT_HAND_LANDMARKS").Get<NormalizedLandmarkList>();
    json_data["RHand"] = HandLandmarksToJson(right_hand_landmarks, ratio);
} else {
    json_data["RHand"] = nullptr; // None equivalent in JSON
}

if (!cc->Inputs().Tag("FACE_LANDMARKS").IsEmpty()) {
    const auto& face_landmarks = cc->Inputs().Tag("FACE_LANDMARKS").Get<NormalizedLandmarkList>();
    json_data["Face"] = FaceLandmarksToJson(face_landmarks, ratio, FACE_LANDMARKS);
} else {
    json_data["Face"] = nullptr; // None equivalent in JSON
}

if (!cc->Inputs().Tag("FACE_BLENDSHAPES").IsEmpty()) {
    const auto& blendshapes = cc->Inputs().Tag("FACE_BLENDSHAPES").Get<ClassificationList>();
    json_data["BlendShape"] = ClassificationListToJson(blendshapes);
} else {
    json_data["BlendShape"] = nullptr; // None equivalent in JSON
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

    server_addr.sin_port = htons(absl::GetFlag(FLAGS_port));
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
        // std::cout << "Sent " << send_result << " bytes to UDP port 12500." << std::endl;
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

float clip(float n, float lower, float upper) {
    return std::max(lower, std::min(n, upper));
}

nlohmann::json FaceBlendshapesPrinter::ClassificationListToJson(const ClassificationList& classification_list) const {
    std::vector<float> values;

    // Start from the second element and populate the `values` vector.
    for (size_t i = 1; i < classification_list.classification_size(); ++i) {
        values.push_back(classification_list.classification(i).score());
    }

    // Define the pairs of indices to swap based on starting from index 1 of classification_list.
    std::vector<std::pair<int, int>> swap_pairs = {
        {0, 1}, {3, 4}, {6, 7}, {8, 9}, {10, 11},
        {12, 13}, {14, 15}, {16, 17}, {18, 19}, {20, 21},
        {23, 25}, {27, 28}, {29, 30}, {32, 38}, {33, 34},
        {35, 36}, {43, 44}, {45, 46}, {47, 48}, {49, 50}
    };

    // Swap the elements in the vector based on the adjusted index pairs.
    for(auto& pair : swap_pairs) {
        std::swap(values[pair.first], values[pair.second]);
    }

        std::vector<float> a = {
        1.8, 2.0, 1.0, 2.5, 1.3, 1.0, 1.0, 1.0, 1.3, 1.3, 
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.3, 1.6, 
        3.0, 1.5, 1.0, 0.8, 1.3, 0.8, 1.5, 1.0, 1.0, 2.5, 
        2.5, 1.3, 0.8, 1.0, 2.0, 2.0, 2.0, 1.2, 0.8, 0.6,
        0.6, 1.4, 10, 2.0, 2.0, 0.3, 0.3, 7.0, 7.0, 0.0, 
        0.0, 1.0
    };

    std::vector<float> b = {
        0.0, 0.0, 0.0, -0.5, -0.3, 0.0, 0.0, 0.0, -0.2, -0.2, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2, -0.2, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, -0.2, 0.0, 0.0,
        0.0, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0
    };

    // Apply the adjustment and clip the results
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = clip(values[i] * a[i] + b[i], 0.0f, 0.9f);
    }

    // Optionally, debug print to ensure swapping worked
    // std::cout << "Blendshape swapped: " << values[50] << std::endl;

    // Convert `values` to a JSON array and return
    nlohmann::json json_values = values;
    return json_values;
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
