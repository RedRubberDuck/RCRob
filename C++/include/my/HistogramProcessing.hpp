#ifndef HISTOGRAM_PROCESSING_HPP
#define HISTOGRAM_PROCESSING_HPP

#include <opencv2/opencv.hpp>

#include <boost/python.hpp>
#include <boost/container/vector.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>

#include <iostream>
#include <cmath>



namespace my{
    using namespace boost::python;

    using Point_t = std::complex<float>;
    typedef std::vector<Point_t> PointVector_t;



    class HistogramProcessing{
        public:
            HistogramProcessing(float,float,uint,uint,uint,float);
        
            PointVector_t apply(cv::Mat,float);
            cv::Mat getKernel();
        private:
            const uint m_width,m_height;
            const float m_inferiorLimitSize;
            const float m_superiorLimitSize;
            const uint  m_lineThinkness;
            const float m_xDistanceLimit;
            cv::Mat m_kernel;
            cv::Point m_anchor;


            

    };
    

    class VideoViewer{
        public:
            VideoViewer(uint,uint);
            virtual void view(cv::Mat);
        private:
            const cv::Size m_size;
    };

    class VideoViewerWrapper:public VideoViewer{
        public:
            VideoViewerWrapper(uint,uint);
            void view(PyObject*);
    };
};


#endif