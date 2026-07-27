cmake_minimum_required(VERSION 3.22.1)
project(yolo_jni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Align with Android 15+ 16 KB page-size requirement.
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")

# MNN headers are vendored in third_party/; the prebuilt libMNN.so ships in
# src/main/jniLibs/arm64-v8a (packaged into the AAR automatically).
set(MNN_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../third_party/mnn/include")
set(MNN_LIB "${CMAKE_CURRENT_LIST_DIR}/../jniLibs/${ANDROID_ABI}/libMNN.so")

add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES
    IMPORTED_LOCATION "${MNN_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${MNN_INCLUDE_DIR}")

add_library(yolo_jni SHARED yolo_jni.cpp)
target_link_libraries(yolo_jni PRIVATE MNN log android dl)
