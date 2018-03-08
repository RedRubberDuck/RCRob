#include <ctime>
#include <fstream>
#include <iostream>
#include <raspicam/raspicam.h>
#include <unistd.h> // for usleep()

using namespace std;

#define NFRAMES 1000
#define WIDTH   1280
#define HEIGHT  960

int main ( int argc,char **argv ) {

    raspicam::RaspiCam Camera;
    // Allowable values: RASPICAM_FORMAT_GRAY,RASPICAM_FORMAT_RGB,RASPICAM_FORMAT_BGR,RASPICAM_FORMAT_YUV420
    Camera.setFormat(raspicam::RASPICAM_FORMAT_GRAY);

    // Allowable widths: 320, 640, 1280
    // Allowable heights: 240, 480, 960
    // setCaptureSize(width,height)
    Camera.setCaptureSize(WIDTH,HEIGHT);

    // Open camera 
    cout<<"Opening Camera..."<<endl;
    if ( !Camera.open()) {cerr<<"Error opening camera"<<endl;return -1;}

    // Wait until camera stabilizes
    cout<<"Sleeping for 3 secs"<<endl;
    usleep(3000000);
    cout << "Grabbing " << NFRAMES << " frames" << endl;

    // Allocate memory for camera buffer
    unsigned long bytes=Camera.getImageBufferSize();
    cout << "Width: "  << Camera.getWidth() << endl;
    cout << "Height: " << Camera.getHeight() << endl;
    cout << "ImageBufferSize: " << bytes << endl;;
    unsigned char *data=new unsigned char[bytes];

    for(int frame=0;frame<NFRAMES;frame++){
       // Capture frame
       Camera.grab();

       // Extract the image
       Camera.retrieve (data,raspicam::RASPICAM_FORMAT_IGNORE);
       }
    }
    return 0;
}