#include "net.h"
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cstdio>
#include <fstream>

static const char* class_names[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "사용법: %s [비디오 경로 또는 0(웹캠)]\n", argv[0]);
        return -1;
    }

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    if (net.load_param("yolov8n.param") != 0 || net.load_model("yolov8n.bin") != 0)
    {
        fprintf(stderr, "전체 모델 파일(yolov8n.param, yolov8n.bin)을 로드할 수 없습니다.\n");
        return -1;
    }

    cv::VideoCapture cap;
    if (std::string(argv[1]) == "0") cap.open(0);
    else cap.open(argv[1]);

    if (!cap.isOpened())
    {
        fprintf(stderr, "비디오를 열 수 없습니다: %s\n", argv[1]);
        return -1;
    }

    const int target_size = 640;
    float conf_threshold = 0.5f;
    int frame_count = 0;

    // --- 중요: Netron으로 확인한 실제 텐서 이름으로 수정해야 합니다. ---
    const char* intermediate_tensor_name = "/model.4/Concat_output_0";

    cv::Mat frame;
    while (true)
    {
        double t_start_total = (double)cv::getTickCount();

        cap >> frame;
        if (frame.empty())
        {
            printf("비디오의 끝에 도달했습니다. 종료합니다.\n");
            break;
        }

        // --- 전처리 ---
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(frame.data, ncnn::Mat::PIXEL_BGR, frame.cols, frame.rows, targe>
        const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
        in.substract_mean_normalize(0, norm_vals);

        // ==========================================================
        // ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 1부: Front-end 추론 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        // ==========================================================
        double t_start_front = (double)cv::getTickCount();
        ncnn::Mat intermediate_tensor;
        {
            ncnn::Extractor ex = net.create_extractor();
            ex.input("images", in);
            ex.extract(intermediate_tensor_name, intermediate_tensor);
        }

        // 중간 텐서를 파일에 저장
        std::ofstream outfile("intermediate.bin", std::ios::binary);
        int dims[4] = { intermediate_tensor.w, intermediate_tensor.h, intermediate_tensor.c, (int)intermediate_tenso>
        outfile.write((char*)dims, sizeof(dims));
        outfile.write((char*)intermediate_tensor.data, intermediate_tensor.total() * dims[3]);
        outfile.close();
        double t_end_front = (double)cv::getTickCount();

        // ==========================================================
        // ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 2부: Back-end 추론 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        // ==========================================================
        double t_start_back = (double)cv::getTickCount();
        ncnn::Mat loaded_tensor;

        // 파일에서 중간 텐서 로드
        std::ifstream infile("intermediate.bin", std::ios::binary);
        int loaded_dims[4];
        infile.read((char*)loaded_dims, sizeof(loaded_dims));
        loaded_tensor.create(loaded_dims[0], loaded_dims[1], loaded_dims[2], (size_t)loaded_dims[3]);
        infile.read((char*)loaded_tensor.data, loaded_tensor.total() * loaded_dims[3]);
        infile.close();

        ncnn::Mat final_out;
        {
            ncnn::Extractor ex = net.create_extractor();
            ex.input(intermediate_tensor_name, loaded_tensor);
            ex.extract("output0", final_out);
        }
        double t_end_back = (double)cv::getTickCount();

        const int num_proposals = final_out.w;
        const int num_classes = final_out.h - 4;
        size_t detected_count = 0;
        for (int i = 0; i < num_proposals; i++)
        {
            float max_score = final_out.row(4)[i];
            for (int j = 1; j < num_classes; j++) {
                if (final_out.row(4 + j)[i] > max_score) {
                    max_score = final_out.row(4 + j)[i];
                }
            }
            if (max_score > conf_threshold) {
                detected_count++;
            }
        }

        double t_end_total = (double)cv::getTickCount();

        double tick_freq = cv::getTickFrequency();
        double total_time_ms = (t_end_total - t_start_total) / tick_freq * 1000.0;
        double front_time_ms = (t_end_front - t_start_front) / tick_freq * 1000.0;
        double back_time_ms = (t_end_back - t_start_back) / tick_freq * 1000.0;
        double fps = (total_time_ms > 0) ? (1000.0 / total_time_ms) : 0;

        frame_count++;
        printf("Frame %4d | FPS: %5.1f | Total: %6.1f ms (Front+Save: %5.1f ms, Load+Back: %5.1f ms) | Detected: %zu>
               frame_count, fps, total_time_ms, front_time_ms, back_time_ms, detected_count);
    }

    cap.release();
    return 0;
}
