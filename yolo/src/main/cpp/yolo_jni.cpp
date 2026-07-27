//
//  yolo_jni.cpp
//  MNN Interpreter API bridge for YOLO11 detection (single input / single output).
//
//  Mirrors the inference pattern proven in the PP-OCR MNN bridge: Interpreter +
//  createSession, resizeTensor/resizeSession for the fixed [1,3,640,640] input,
//  and a Tensor(t, CAFFE) host-tensor round-trip for BOTH input
//  (copyFromHostTensor) and output (copyToHostTensor). Reading the output
//  straight from readMap()/host() returns MNN's internal packed layout and
//  scrambles the [1,84,8400] detection tensor — the CAFFE round-trip is
//  mandatory.
//

#include <jni.h>
#include <android/log.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/MNNForwardType.h>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {
struct Runtime {
    std::unique_ptr<MNN::Interpreter> interp;
    MNN::Session* session = nullptr;
    std::mutex mutex;
};

Runtime* runtime(jlong handle) { return reinterpret_cast<Runtime*>(handle); }

std::string fromJString(JNIEnv* env, jstring value) {
    if (!value) return {};
    const char* chars = env->GetStringUTFChars(value, nullptr);
    std::string result = chars ? chars : "";
    if (chars) env->ReleaseStringUTFChars(value, chars);
    return result;
}

void throwError(JNIEnv* env, const std::string& message) {
    jclass clazz = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(clazz, message.c_str());
}

// CPU-only schedule (the Adreno OpenCL driver crashes in clEnqueueRecordingQCOM).
MNN::Session* createCpuSession(MNN::Interpreter* interp, int threads) {
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.backupType = MNN_FORWARD_CPU;
    config.numThread = threads;
    MNN::BackendConfig backendConfig;
    backendConfig.power = MNN::BackendConfig::Power_High;
    backendConfig.memory = MNN::BackendConfig::Memory_Normal;
    backendConfig.precision = MNN::BackendConfig::Precision_Normal;
    config.backendConfig = &backendConfig;
    return interp->createSession(config);
}
}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_yolomnn_YoloDetector_nativeInit(JNIEnv* env, jclass) {
    try {
        return reinterpret_cast<jlong>(new Runtime());
    } catch (const std::exception& error) {
        throwError(env, std::string("nativeInit failed: ") + error.what());
    }
    return 0;
}

extern "C" JNIEXPORT void JNICALL
Java_com_yolomnn_YoloDetector_nativeLoad(
        JNIEnv* env, jclass, jlong handle, jstring modelPath, jint threads) {
    auto* state = runtime(handle);
    if (!state) { throwError(env, "Invalid runtime handle"); return; }
    const std::string path = fromJString(env, modelPath);
    if (path.empty()) { throwError(env, "model path is empty"); return; }
    std::lock_guard<std::mutex> lock(state->mutex);
    // Release any previously loaded model (supports switching n <-> s).
    if (state->session) {
        state->interp->releaseSession(state->session);
        state->session = nullptr;
    }
    state->interp.reset(MNN::Interpreter::createFromFile(path.c_str()));
    if (!state->interp) { throwError(env, std::string("createFromFile failed: ") + path); return; }
    state->session = createCpuSession(state->interp.get(), threads > 0 ? threads : 4);
    if (!state->session) { throwError(env, "createSession failed"); return; }
}

// Run the detector on a [1,3,inH,inW] NCHW float tensor. Returns a flat float[]
// = [ ...output..., outDim1, outDim2 ] where the trailing two values are the
// last two dimensions of the output tensor (e.g. 84, 8400) so Kotlin can
// reshape without hardcoding.
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_yolomnn_YoloDetector_nativeRun(
        JNIEnv* env, jclass, jlong handle,
        jfloatArray jInput, jint inH, jint inW) {
    auto* state = runtime(handle);
    if (!state || !state->session) return nullptr;
    std::lock_guard<std::mutex> lock(state->mutex);
    jfloat* inPtr = nullptr;
    try {
        jsize inLen = env->GetArrayLength(jInput);
        inPtr = env->GetFloatArrayElements(jInput, nullptr);
        if (!inPtr) return nullptr;

        auto inMap = state->interp->getSessionInputAll(state->session);
        if (inMap.empty()) throw std::runtime_error("no input tensor");
        MNN::Tensor* input = inMap.begin()->second;
        state->interp->resizeTensor(input, {1, 3, inH, inW});
        state->interp->resizeSession(state->session);
        inMap = state->interp->getSessionInputAll(state->session);
        input = inMap.begin()->second;

        std::unique_ptr<MNN::Tensor> inputHost(new MNN::Tensor(input, MNN::Tensor::CAFFE));
        if (inputHost->elementSize() != static_cast<int>(inLen)) {
            throw std::runtime_error("input size mismatch: host=" +
                std::to_string(inputHost->elementSize()) + " got=" + std::to_string(inLen));
        }
        std::memcpy(inputHost->host<float>(), inPtr, static_cast<size_t>(inLen) * sizeof(float));
        input->copyFromHostTensor(inputHost.get());

        env->ReleaseFloatArrayElements(jInput, inPtr, 0);
        inPtr = nullptr;

        if (state->interp->runSession(state->session) != MNN::NO_ERROR) {
            throw std::runtime_error("runSession failed");
        }

        auto outMap = state->interp->getSessionOutputAll(state->session);
        if (outMap.empty()) throw std::runtime_error("no output tensor");
        MNN::Tensor* output = outMap.begin()->second;
        std::vector<int> outShape = output->shape();
        std::unique_ptr<MNN::Tensor> outputHost(new MNN::Tensor(output, MNN::Tensor::CAFFE));
        output->copyToHostTensor(outputHost.get());

        const float* vals = outputHost->host<float>();
        size_t total = static_cast<size_t>(outputHost->elementSize());
        // Last two dims (channels, anchors) for reshape on the Kotlin side.
        int d1 = outShape.size() >= 2 ? outShape[outShape.size() - 2] : 0;
        int d2 = outShape.size() >= 1 ? outShape[outShape.size() - 1] : 0;

        jfloatArray result = env->NewFloatArray(static_cast<jsize>(total) + 2);
        if (!result) return nullptr;
        jfloat* dst = env->GetFloatArrayElements(result, nullptr);
        std::memcpy(dst, vals, total * sizeof(float));
        dst[total]     = static_cast<float>(d1);
        dst[total + 1] = static_cast<float>(d2);
        env->ReleaseFloatArrayElements(result, dst, 0);
        return result;
    } catch (const std::exception& error) {
        if (inPtr) env->ReleaseFloatArrayElements(jInput, inPtr, 0);
        throwError(env, std::string("nativeRun failed: ") + error.what());
    } catch (...) {
        if (inPtr) env->ReleaseFloatArrayElements(jInput, inPtr, 0);
        throwError(env, "nativeRun failed with an unknown error");
    }
    return nullptr;
}

extern "C" JNIEXPORT void JNICALL
Java_com_yolomnn_YoloDetector_nativeRelease(JNIEnv*, jclass, jlong handle) {
    if (handle != 0) delete runtime(handle);
}
