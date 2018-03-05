#ifndef MY_PREPROCESS_HPP
#define MY_PREPROCESS_HPP

#include <opencv2/opencv.hpp>
#include <string>


namespace my{


    class Filter{
        public:
            Filter(int);
            virtual cv::Mat operator()(const cv::Mat&);
        private:
            const int           m_kernelSize;

    };


    class PerspectiveTransform{
        public:
            PerspectiveTransform(cv::Mat,cv::Mat,cv::Size2i,float);

            void transfor2BirdView(const cv::Mat&,cv::Mat&);


            static PerspectiveTransform init2();
        private:
            const cv::Mat           m_M,m_M_inv;
            const cv::Size2i        m_size;
            const float             m_pxpcm;

    };
};
#endif