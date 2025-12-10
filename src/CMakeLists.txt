# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 14)

if(MSVC)
	add_subdirectory(dll)
else()
	#file(GLOB CPP_SOURCES *.cpp)
	set(CPP_SOURCES ann_exception.cpp aux_utils.cpp index.cpp
        linux_aligned_file_reader.cpp math_utils.cpp memory_mapper.cpp
        partition_and_pq.cpp pq_flash_index.cpp logger.cpp distance.cpp
		 utils.cpp v2/graph_delta.cpp v2/index_merger.cpp 
        v2/merge_insert.cpp)
	add_library(${PROJECT_NAME} ${CPP_SOURCES})
	add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
    
    if(DISKANN_MKL_LIBS)
        target_link_libraries(${PROJECT_NAME} PUBLIC ${DISKANN_MKL_LIBS} pthread m dl)
        target_link_libraries(${PROJECT_NAME}_s PUBLIC ${DISKANN_MKL_LIBS} pthread m dl)
    else()
        target_link_libraries(${PROJECT_NAME} PUBLIC pthread m dl)
        target_link_libraries(${PROJECT_NAME}_s PUBLIC pthread m dl)
    endif()
endif()
install()
