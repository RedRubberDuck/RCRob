#include <stdio.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>



int
main(int argc, char *argv[])
{

    std::cout<<"Start Main"<<std::endl;
	std::string folder ("");
	std::string fileName ("/home/nandi/Workspaces/git/resource/videos/martie2/test11.h264");

	cv::VideoCapture cap(folder+fileName);
	if(!cap.isOpened()){  // check if we succeeded
		std::cout<<"The file not found! "+folder+fileName<<std::endl;
		return -1;
	}

    cv::Mat frame;
    cv::namedWindow("",1);

    cv::Size size;
    size.height=400;;
    size.width=600;

    int index=0;

    while(cap.read(frame)){
        
        cv::Mat res;
        cv::resize(frame,res,size);

        cv::imshow("",res);
		int key = cv::waitKey(33);
        if(key == 'q'){
            break;
        }

        if(index==100){
            break;
        }
        index++;
    }
    
}
