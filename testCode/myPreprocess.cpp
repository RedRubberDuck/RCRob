#include "myPreprocess.hpp"

my::Filter::Filter(int f_kernelSize)
        :m_kernelSize(f_kernelSize)
{
}


cv::Mat  my::Filter::operator()(const  cv::Mat& f_input){
        cv::Mat l_bwdst;
        cv::cvtColor(f_input,l_bwdst,cv::COLOR_RGB2GRAY);
        cv::Mat l_mask;
        cv::adaptiveThreshold(l_bwdst,l_mask,255,cv::ADAPTIVE_THRESH_MEAN_C,cv::THRESH_BINARY,m_kernelSize,-8.5);

        return l_mask;
}



my::PerspectiveTransform::PerspectiveTransform(cv::Mat f_M,cv::Mat f_M_inv,cv::Size2i f_size,float f_pxpcm)
                :m_M(f_M)
                ,m_M_inv(f_M_inv)
                ,m_size(f_size)
                ,m_pxpcm(f_pxpcm)
{
}


my::PerspectiveTransform my::PerspectiveTransform::init2(){
        
        cv:: Point2f inputQuad[4];
        cv:: Point2f outputQuad[4];

        inputQuad[0] = cv::Point2f(421,214);
        inputQuad[1] = cv::Point2f(1354,188);
        inputQuad[2] = cv::Point2f(-295,609);
        inputQuad[3] = cv::Point2f(2131,572);

        float pxpcm = 2.0;
        float step = 45.0;

        outputQuad[0] = cv::Point2f(0,0);
        outputQuad[1] = cv::Point2f(2*step*pxpcm,0);
        outputQuad[2] = cv::Point2f(0,2*step*pxpcm);
        outputQuad[3] = cv::Point2f(2*step*pxpcm,2*step*pxpcm);
        cv::Mat l_M( 2, 4, CV_32FC1 ), l_M_inv( 2, 4, CV_32FC1 );

        l_M = getPerspectiveTransform( inputQuad, outputQuad );
        l_M_inv = getPerspectiveTransform(outputQuad, inputQuad);
        cv::Size2i l_size(2*step*pxpcm,2*step*pxpcm);

        return my::PerspectiveTransform(l_M,l_M_inv,l_size,pxpcm);
}

void my::PerspectiveTransform::transfor2BirdView(const cv::Mat& f_src, cv::Mat& f_dst){
        cv::warpPerspective(f_src,f_dst,m_M,m_size);
}