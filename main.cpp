#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;

const int connect_list[36] = { 0, 1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 6, 5, 7, 7, 9, 6, 8, 8, 10, 5, 11, 6, 12, 11, 12, 11, 13, 13, 15, 12, 14, 14, 16 };

class E2Pose
{
public:
	E2Pose(string model_path, float confThreshold);
	void detect(Mat& cv_image);
private:
	float confThreshold;

	void normalize_(Mat img);
	int inpWidth;
	int inpHeight;
	vector<float> input_image_;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "E2Pose");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

E2Pose::E2Pose(string model_path, float confThreshold)
{
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	this->confThreshold = confThreshold;
}

void E2Pose::normalize_(Mat img)
{
	//img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + c];
				this->input_image_[c * row * col + i * col + j] = pix;
			}
		}
	}
}

void E2Pose::detect(Mat& srcimg)
{
	Mat dstimg;
	cvtColor(srcimg, dstimg, COLOR_BGR2RGB);
	resize(dstimg, dstimg, Size(this->inpWidth, this->inpHeight));
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());

	const float* kpt = ort_outputs[0].GetTensorMutableData<float>();
	const float* pv = ort_outputs[1].GetTensorMutableData<float>();
	const int num_proposal = ort_outputs.at(0).GetTensorTypeAndShapeInfo().GetShape().at(1);
	const int num_pts = ort_outputs.at(0).GetTensorTypeAndShapeInfo().GetShape().at(2);
	const int len = num_pts * 3;

	vector<vector<int>> results;
	for (int i = 0; i < num_proposal; i++)
	{
		if (pv[i] >= this->confThreshold)
		{
			vector<int> human_pts(num_pts * 2, 0);
			for (int j = 0; j < num_pts; j++)
			{
				const float score = kpt[j * 3] * 2;
				if (score >= this->confThreshold)
				{
					const float x = kpt[j * 3 + 1] * srcimg.cols;
					const float y = kpt[j * 3 + 2] * srcimg.rows;
					human_pts[j * 2] = int(x);
					human_pts[j * 2 + 1] = int(y);
				}
			}
			results.push_back(human_pts);
		}
		kpt += len;
	}

	for (size_t i = 0; i < results.size(); ++i)
	{
		for (int j = 0; j < num_pts; j++)
		{
			const int cx = results[i][j * 2];
			const int cy = results[i][j * 2 + 1];
			if (cx > 0 && cy > 0)
			{
				circle(srcimg, Point(cx, cy), 3, Scalar(0, 0, 255), -1, LINE_AA);
			}
			
			const int start_x = results[i][connect_list[j * 2] * 2];
			const int start_y = results[i][connect_list[j * 2] * 2 + 1];
			const int end_x = results[i][connect_list[j * 2 + 1] * 2];
			const int end_y = results[i][connect_list[j * 2 + 1] * 2 + 1];
			if (start_x > 0 && start_y > 0 && end_x > 0 && end_y > 0)
			{
				line(srcimg, Point(start_x, start_y), Point(end_x, end_y), Scalar(0, 255, 0), 2, LINE_AA);
			}
		}
		const int start_x = results[i][connect_list[num_pts * 2] * 2];
		const int start_y = results[i][connect_list[num_pts * 2] * 2 + 1];
		const int end_x = results[i][connect_list[num_pts * 2 + 1] * 2];
		const int end_y = results[i][connect_list[num_pts * 2 + 1] * 2 + 1];
		if (start_x > 0 && start_y > 0 && end_x > 0 && end_y > 0)
		{
			line(srcimg, Point(start_x, start_y), Point(end_x, end_y), Scalar(0, 255, 0), 2, LINE_AA);
		}
	}
}

int main()
{
	E2Pose mynet("weights/e2epose_resnet50_1x3x512x512.onnx", 0.5);
	string imgpath = "images/person.jpg";
	Mat srcimg = imread(imgpath);
	mynet.detect(srcimg);

	static const string kWinName = "Deep learning object detection in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}