#ifndef SLICING_IMAGE_HPP
#define SLICING_IMAGE_HPP

// #include <boost/lockfree/queue.hpp>
#include <opencv2/opencv.hpp>

#include <my/HistogramProcessing.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread.hpp>
#include <boost/lockfree/queue.hpp>

#define NR_WORKERS 2


namespace my{
    using FramePair  =  std::pair<cv::Mat,float>;
    using FrameQueue =  std::vector<FramePair>;

    class Worker{
        public:
            Worker(FrameQueue&,boost::mutex&,float,float,uint,uint,uint,float);

            void run();
            PointVector_t getPoints();
        private:
            FrameQueue&                 m_frameBuffer;
            boost::mutex&               m_mtx;
            my::HistogramProcessing     m_histogramProcessing;
            PointVector_t               m_pointVec;
                          

    };


    class SlicingMethod{
        public:
            SlicingMethod(uint,float,float,uint,uint,uint,float);
            ~SlicingMethod();

            PointVector_t apply(cv::Mat);

        private:
            const uint                          m_nrSlices;
            // const uint                          m_nrWorkers; 
            
            FrameQueue                          m_frameBuffer;
            boost::mutex                        m_mutex;
            std::array<Worker*,NR_WORKERS>      m_workers;

    };


};



#endif