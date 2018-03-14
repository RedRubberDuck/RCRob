#include <my/SlicingImage.hpp>


my::Worker::Worker( FrameQueue&             f_frameBruffer
                    ,boost::mutex&          f_mtx
                    ,float                  f_inferiorRate
                    ,float                  f_superiorRate
                    ,uint                   f_width
                    ,uint                   f_height
                    ,uint                   f_lineThinkness
                    ,float                  f_xDistanceLimit)
            :m_frameBuffer(f_frameBruffer)
            ,m_mtx(f_mtx)
            ,m_histogramProcessing(f_inferiorRate,f_superiorRate,f_width,f_height,f_lineThinkness,f_xDistanceLimit)
{
}

void my::Worker::run(){
    m_pointVec.clear();
    m_mtx.lock();
    while(m_frameBuffer.size()>0){
        my::FramePair l_framePair;
        try{
            l_framePair = m_frameBuffer.back();
            m_frameBuffer.pop_back();
            m_mtx.unlock();
        }catch(...){
            std::cout<<"Exception";
        }
    
        //To do something
        my::PointVector_t l_vec = m_histogramProcessing.apply(l_framePair.first,l_framePair.second);
        m_pointVec.insert(m_pointVec.end(),l_vec.begin(),l_vec.end());
        m_mtx.lock(); 
    }
    m_mtx.unlock();
}


my::PointVector_t my::Worker::getPoints(){
    return m_pointVec;
}

my::SlicingMethod::SlicingMethod(   uint                    f_nrSlices
                                    ,float                  f_inferiorRate
                                    ,float                  f_superiorRate
                                    ,uint                   f_width
                                    ,uint                   f_height
                                    ,uint                   f_lineThinkness
                                    ,float                  f_xDistanceLimit)
                                    :m_nrSlices(f_nrSlices)
                                    ,m_frameBuffer()
                                    // ,m_mutex()
                                    ,m_workers()
{


    for ( uint worker_i=0 ; worker_i < NR_WORKERS ; worker_i++){
        Worker* l_ptr_worker=new Worker(m_frameBuffer,m_mutex,f_inferiorRate,f_superiorRate,f_width,f_height,f_lineThinkness,f_xDistanceLimit);
        m_workers[worker_i] = l_ptr_worker;
    }

}


my::SlicingMethod::~SlicingMethod(){
    for ( uint worker_i=0 ; worker_i < NR_WORKERS ; worker_i++){
        delete m_workers[worker_i];
    }
}


my::PointVector_t my::SlicingMethod::apply(      cv::Mat         f_frame){
    my::PointVector_t l_allPoints;
    try{
        cv::Size l_size = f_frame.size();
        uint l_stepLength = l_size.height/m_nrSlices;
        for( uint slice_i = 0 ; slice_i < m_nrSlices ; slice_i++ ){
            cv::Mat l_slice = f_frame(cv::Range(l_stepLength*slice_i,l_stepLength*(slice_i+1)),cv::Range::all());
            float l_y = l_size.height*(slice_i+0.5)/m_nrSlices;
            FramePair l_framePair(l_slice,l_y);
            m_frameBuffer.push_back(l_framePair);
        }

        std::array<boost::thread*,NR_WORKERS> l_workersThread; 
        for( uint worker_i = 0 ; worker_i < NR_WORKERS ; worker_i++){
            boost::thread* l_workerThread=new boost::thread(&my::Worker::run,m_workers[worker_i]);
            l_workersThread[worker_i]=l_workerThread;
        }
    
        for( uint worker_i = 0 ; worker_i < NR_WORKERS ; worker_i++){
            l_workersThread[worker_i]->join();
            my::PointVector_t l_point = m_workers[worker_i] -> getPoints();
            l_allPoints.insert( l_allPoints.end(),l_point.begin(),l_point.end());
            delete l_workersThread[worker_i];
        }
    }catch(...){
        std::cout<<"Exception"<<std::endl;
    }
    return l_allPoints;
}