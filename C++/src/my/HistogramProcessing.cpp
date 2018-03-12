#include <my/HistogramProcessing.hpp>



my::HistogramProcessing::HistogramProcessing(   float               f_inferiorRate
                                                ,float              f_superiorRate
                                                ,uint               f_width
                                                ,uint               f_height
                                                ,uint              f_lineThinkness
                                                ,float              f_xDistanceLimit)
                                :m_width(f_width)
                                ,m_height(f_height)
                                ,m_inferiorLimitSize(f_width*f_height*f_inferiorRate)
                                ,m_superiorLimitSize(f_width*f_height*f_superiorRate)
                                ,m_lineThinkness(f_lineThinkness)
                                ,m_xDistanceLimit(f_xDistanceLimit)
{

    uint l_kernel_size;
    if(f_lineThinkness%2==0){
        l_kernel_size = f_lineThinkness+1;
    }else{
        l_kernel_size = f_lineThinkness;
    }

    m_kernel = cv::Mat::ones(1,l_kernel_size,CV_32F)/l_kernel_size;
    m_anchor = cv::Point( -1, -1 );
}


my::PointVector_t my::HistogramProcessing::apply(cv::Mat l_img_part,float f_pointY){
    cv::Mat l_hist;
    cv::Mat l_img_part_float;
    l_img_part.assignTo(l_img_part_float,CV_32F);
    cv::reduce(l_img_part_float,l_hist,0,cv::REDUCE_SUM);

    cv::Mat l_hist_ftl;
    cv::filter2D(l_hist,l_hist_ftl,-1,m_kernel,m_anchor);
    
    PointVector_t l_points;

    float l_sum = 0;
    cv::Size l_size = l_hist_ftl.size();
    uint l_start_px=0;
    
    for(uint col_i=1; col_i < l_size.width ;col_i++){
    
        if(l_hist_ftl.at<float>(0,col_i)  > 0 ){
            if(l_hist_ftl.at<float>(0,col_i-1) == 0.0){
                l_start_px = col_i;
            }
            l_sum += l_hist_ftl.at<float>(0,col_i);
        }
        else if( l_hist_ftl.at<float>(0,col_i)  == 0.0 && l_hist_ftl.at<float>(0,col_i-1)  > 0.0) {
            
            if ( l_sum < m_superiorLimitSize && l_sum > m_inferiorLimitSize  ){
                uint l_pointX = (l_start_px + col_i )/2;
                if(l_points.size()>0 && abs(l_points[l_points.size()-1].real() - l_pointX ) < m_xDistanceLimit ){
                    Point_t l_point( (l_points[l_points.size()-1].real() + l_pointX)/2 ,f_pointY);
                    l_points.push_back(l_point);
                }else{
                    Point_t l_point(l_pointX,f_pointY);
                    l_points.push_back(l_point);
                }
            }
            
            l_sum  = 0;
        }else{
            l_sum  = 0;
        }
    }

    return l_points;
}

cv::Mat my::HistogramProcessing::getKernel(){
    return m_kernel;
}







my::VideoViewer::VideoViewer(uint f_width, uint f_height)
    :m_size(f_width,f_height)
{
}

void my::VideoViewer::view(cv::Mat frame){
    cv::Mat l_frame_resized;
    cv::resize(frame,l_frame_resized,m_size);

    cv::imshow("",l_frame_resized);
    cv::waitKey();
}

my::VideoViewerWrapper::VideoViewerWrapper(uint f_width,uint f_height)
    :my::VideoViewer(f_width,f_height)
{}

void my::VideoViewerWrapper::view(PyObject* l_obj_frame){
    cv::Mat l_frame;
    l_frame = pbcvt::fromNDArrayToMat(l_obj_frame);
    my::VideoViewer::view(l_frame);
}


