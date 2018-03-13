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
    
    
    // 
    // bool run=false;

    // do{
    //     m_mtx.lock();
        
    //     m_mtx.unlock(); 


    // }while(run);
    m_mtx.lock();
    while(m_frameBuffer.size()>0){
        
        // std::cout<<" Lock ";
        try{
            // m_mtx.lock();
            my::FramePair l_framePair;
            l_framePair = m_frameBuffer.back();
            m_frameBuffer.pop_back();
            std::cout<<"Process"<<l_framePair.second<<std::endl;
            m_mtx.unlock();
        }catch(...){
            // m_mtx.unlock();
            std::cout<<"Exception";
        }
        
        // std::cout<<" UnLock ";
        
        
        //To do something
        // my::PointVector_t l_vec = m_histogramProcessing.apply(l_framePair.first,l_framePair.second);
        // m_pointVec.assign(l_vec.begin(),l_vec.end());
        // std::cout<<" Lock ";
        m_mtx.lock(); 
    }
    // std::cout<<" UnLock ";
    m_mtx.unlock();
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


void my::SlicingMethod::apply(      cv::Mat         f_frame){
    cv::Size l_size = f_frame.size();
    uint l_stepLength = l_size.height/m_nrSlices;
    for( uint slice_i = 0 ; slice_i < m_nrSlices ; slice_i++ ){
        cv::Mat l_slice = f_frame(cv::Range(l_stepLength*slice_i,l_stepLength*(slice_i+1)),cv::Range::all());
        FramePair l_framePair(l_slice,l_size.height*(slice_i+0.5));
        m_frameBuffer.push_back(l_framePair);
    }

    std::array<boost::thread*,NR_WORKERS> l_workersThread; 
    for( uint worker_i = 0 ; worker_i < NR_WORKERS ; worker_i++){
        boost::thread* l_workerThread=new boost::thread(&my::Worker::run,m_workers[worker_i]);
        l_workersThread[worker_i]=l_workerThread;
    }

    try{
        for( uint worker_i = 0 ; worker_i < NR_WORKERS ; worker_i++){
            l_workersThread[worker_i]->join();
            delete l_workersThread[worker_i];
            std::cout<<"Terminate"<<worker_i<<std::endl;
        }
    }catch(...){
        std::cout<<"Exception"<<std::endl;
    }

}