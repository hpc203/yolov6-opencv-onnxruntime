#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	string modelpath;
};

class YOLOV6
{
public:
	YOLOV6(Net_config config);
	void detect(Mat& frame);
private:
	const int inpWidth = 640;
	const int inpHeight = 640;
	vector<string> class_names;
	int num_class;
	
	float confThreshold;
	float nmsThreshold;
	const bool keep_ratio = true;
	Net net;
	void drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid);
	Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);
};

YOLOV6::YOLOV6(Net_config config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;

	this->net = readNet(config.modelpath);
	ifstream ifs("coco.names");
	string line;
	while (getline(ifs, line)) this->class_names.push_back(line);
	this->num_class = class_names.size();
}

Mat YOLOV6::resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 114);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 114);
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

void YOLOV6::drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid)   // Draw the predicted bounding box
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 2);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	label = this->class_names[classid] + ":" + label;

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
}

void YOLOV6::detect(Mat& frame)
{
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);
	Mat blob = blobFromImage(dstimg, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

	int num_proposal = outs[0].size[0];
	int nout = outs[0].size[1];
	if (outs[0].dims > 2)
	{
		num_proposal = outs[0].size[1];
		nout = outs[0].size[2];
		outs[0] = outs[0].reshape(0, num_proposal);
	}
	/////generate proposals
	vector<float> confidences;
	vector<Rect> boxes;
	vector<int> classIds;
	float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;
	int n = 0, row_ind = 0; ///cx,cy,w,h,box_score,class_score
	float* pdata = (float*)outs[0].data;
	for (n = 0; n < num_proposal; n++)   ///ÌØÕ÷Í¼³ß¶È
	{
		float box_score = pdata[4];
		if (box_score > this->confThreshold)
		{
			Mat scores = outs[0].row(row_ind).colRange(5, nout);
			Point classIdPoint;
			double max_class_socre;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
			max_class_socre *= box_score;
			if (max_class_socre > this->confThreshold)
			{ 
				const int class_idx = classIdPoint.x;
				float cx = (pdata[0] - padw) * ratiow;  ///cx
				float cy = (pdata[1] - padh) * ratioh;   ///cy
				float w = pdata[2] * ratiow;   ///w
				float h = pdata[3] * ratioh;  ///h

				int left = int(cx - 0.5 * w);
				int top = int(cy - 0.5 * h);

				confidences.push_back((float)max_class_socre);
				boxes.push_back(Rect(left, top, (int)(w), (int)(h)));
				classIds.push_back(class_idx);
			}
		}
		row_ind++;
		pdata += nout;
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		this->drawPred(confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame, classIds[idx]);
	}
}

int main()
{
	Net_config YOLOV6_nets = { 0.3, 0.5, "onnxmodel/yolov6s.onnx" };
	YOLOV6 YOLOV6_model(YOLOV6_nets);
	string imgpath = "images/image1.jpg";
	Mat srcimg = imread(imgpath);
	YOLOV6_model.detect(srcimg);

	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}