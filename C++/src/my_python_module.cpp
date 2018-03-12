#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>
#include <my/HistogramProcessing.hpp>


namespace my {

    using namespace boost::python;
    typedef std::vector<std::string> Words;

#if (PY_VERSION_HEX >= 0x03000000)

    static void *init_ar() {
#else
        static void init_ar(){
#endif
        Py_Initialize();

        import_array();
        return NUMPY_IMPORT_ARRAY_RETVAL;
    }

    BOOST_PYTHON_MODULE (my) {
        //using namespace XM;
        init_ar();

        //initialize converters
        to_python_converter<cv::Mat,
                pbcvt::matToNDArrayBoostConverter>();
        pbcvt::matFromNDArrayBoostConverter();

        class_<my::VideoViewerWrapper>("VideoViewer",init<uint,uint>())
                                .def("view",&my::VideoViewerWrapper::view);

    

        class_< PointVector_t >("PointVector")
                        .def(vector_indexing_suite< PointVector_t >());

        class_<my::HistogramProcessing>("HistogramProcessing",init<float,float,uint,uint,uint,float>())
                                .def("apply",&my::HistogramProcessing::apply)
                                .def("getKernel",&my::HistogramProcessing::getKernel);

    }

} //end namespace pbcvt
