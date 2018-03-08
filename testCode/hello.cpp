// #include <stdio.h>
// #include <iostream>
// #include <string>
// #include <opencv2/opencv.hpp>
// #include <chrono> 

// #include "myPreprocess.hpp"


int main(int argc, char **argv){
// 	std::cout<<"Start Main"<<std::endl;
// 	std::string folder ("");
// 	std::string fileName ("martie2/test2.h264");

// 	cv::VideoCapture cap(folder+fileName);
// 	if(!cap.isOpened()){  // check if we succeeded
// 		std::cout<<"The file not found! "+folder+fileName<<std::endl;
// 		return -1;
// 	}
	
	
// 	my::Filter l_fiter(11);
// 	my::PerspectiveTransform l_persTrans = my::PerspectiveTransform::init2();
	
// 	cv::Mat frame;
// 	// cv::namedWindow("",1);
// 	std::cout<<"Start to show"<<std::endl;

// 	auto start = std::chrono::high_resolution_clock::now();
// 	double fullelapsed =0;
// 	while(cap.read(frame)){
// 		auto start = std::chrono::high_resolution_clock::now();
// 		cv::Mat birdview; 
// 		l_persTrans.transfor2BirdView(frame,birdview);
// 		cv::Mat bw_img=l_fiter(birdview);
// 		auto finish = std::chrono::high_resolution_clock::now();
// 		std::chrono::duration<double> elapsed = finish - start;
// 		fullelapsed +=  elapsed.count();
// 		std::cout << "Elapsed time: " << elapsed.count() << " s\n";
// 		// cv::imshow("",bw_img);
// 		// cv::waitKey(33);
// 	}
	
	
// 	std::cout << "Elapsed time: " << fullelapsed << " s\n";


	return 0;
}
