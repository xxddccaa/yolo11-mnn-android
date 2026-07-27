package com.yolomnn

import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import kotlin.math.max
import kotlin.math.min

/**
 * On-device YOLO11 detector backed by MNN.
 *
 * The preprocessing / decode follows the reference implementation from
 * wangzhaode/mnn-yolo: letterbox by padding to a max(h,w) square then resizing
 * to 640, /255 normalization (no mean/std), output tensor [1,84,8400] where the
 * 84 channels are [cx, cy, w, h, 80 class probabilities], decoded with a
 * confidence filter, class-agnostic NMS, and box coordinates scaled back to the
 * original image.
 *
 * Native inference happens in [nativeRun]; heavy work must run off the main
 * thread. Instances are [AutoCloseable]; call [close] to free native memory.
 */
class YoloDetector : AutoCloseable {

    private var handle: Long = 0
    private var loadedModelPath: String? = null

    /** Result bundle: detections plus the pure inference time in milliseconds. */
    data class Result(val detections: List<Detection>, val inferenceMs: Long)

    init {
        handle = nativeInit()
        check(handle != 0L) { "YoloDetector nativeInit failed" }
    }

    /** Load (or switch to) an .mnn model file. Safe to call repeatedly. */
    @Synchronized
    fun loadModel(modelPath: String, threads: Int = 4) {
        if (modelPath == loadedModelPath) return
        nativeLoad(handle, modelPath, threads)
        loadedModelPath = modelPath
    }

    /**
     * Detect objects in [bitmap]. Returns detections in bitmap pixel space.
     * [confThreshold] filters low-score boxes; [iouThreshold] controls NMS.
     */
    @Synchronized
    fun detect(
        bitmap: Bitmap,
        confThreshold: Float = 0.25f,
        iouThreshold: Float = 0.45f,
    ): Result {
        val srcW = bitmap.width
        val srcH = bitmap.height
        // Official ultralytics letterbox: scale by the smaller ratio (keep aspect
        // ratio), center the resized image on a 640x640 canvas, pad with gray 114.
        val ratio = min(INPUT_SIZE.toFloat() / srcW, INPUT_SIZE.toFloat() / srcH)
        val newW = Math.round(srcW * ratio)
        val newH = Math.round(srcH * ratio)
        val padX = (INPUT_SIZE - newW) / 2f
        val padY = (INPUT_SIZE - newH) / 2f

        val input = preprocess(bitmap, newW, newH, padX, padY)
        val started = SystemClock.elapsedRealtime()
        val raw = nativeRun(handle, input, INPUT_SIZE, INPUT_SIZE)
            ?: return Result(emptyList(), 0)
        val inferenceMs = SystemClock.elapsedRealtime() - started

        val detections = decode(raw, ratio, padX, padY, confThreshold, iouThreshold, srcW, srcH)
        return Result(detections, inferenceMs)
    }

    /**
     * Official-style letterbox: resize the bitmap to [newW]x[newH] (aspect ratio
     * preserved) and paste it centered (offset [padX],[padY]) onto a 640x640
     * canvas filled with gray 114, then emit an NCHW float array normalized to
     * /255. Bitmap channels are already RGB via getPixels.
     */
    private fun preprocess(bitmap: Bitmap, newW: Int, newH: Int, padX: Float, padY: Float): FloatArray {
        val canvasBmp = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(canvasBmp)
        canvas.drawColor(PAD_COLOR)  // gray 114 padding
        val resized = Bitmap.createScaledBitmap(bitmap, newW, newH, true)
        canvas.drawBitmap(resized, padX, padY, null)
        if (resized !== bitmap) resized.recycle()

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        canvasBmp.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        canvasBmp.recycle()

        val area = INPUT_SIZE * INPUT_SIZE
        val out = FloatArray(3 * area)  // NCHW: R plane, G plane, B plane
        for (i in 0 until area) {
            val p = pixels[i]
            out[i] = ((p shr 16) and 0xFF) / 255f            // R
            out[area + i] = ((p shr 8) and 0xFF) / 255f       // G
            out[2 * area + i] = (p and 0xFF) / 255f           // B
        }
        return out
    }

    /**
     * Decode the flat native output. Layout: [ output(84*8400) ..., d1, d2 ]
     * where d1=84 channels, d2=8400 anchors. Channel order: cx,cy,w,h then 80
     * class scores. Coordinates are in the 640 letterboxed space; undo the
     * letterbox via ((coord - pad) / ratio) and clamp to the original image.
     */
    private fun decode(
        raw: FloatArray,
        ratio: Float,
        padX: Float,
        padY: Float,
        confThreshold: Float,
        iouThreshold: Float,
        srcW: Int,
        srcH: Int,
    ): List<Detection> {
        val d2 = raw[raw.size - 1].toInt()   // anchors, e.g. 8400
        val d1 = raw[raw.size - 2].toInt()   // channels, e.g. 84
        if (d1 < 5 || d2 <= 0) return emptyList()
        val numClasses = d1 - 4

        // raw is channel-major: value at [c, a] = raw[c * d2 + a]
        val candidates = ArrayList<Detection>()
        for (a in 0 until d2) {
            // find best class
            var bestCls = 0
            var bestScore = raw[4 * d2 + a]
            for (c in 1 until numClasses) {
                val s = raw[(4 + c) * d2 + a]
                if (s > bestScore) { bestScore = s; bestCls = c }
            }
            if (bestScore < confThreshold) continue

            // undo letterbox: subtract padding, divide by resize ratio
            val cx = (raw[a] - padX) / ratio
            val cy = (raw[d2 + a] - padY) / ratio
            val w = raw[2 * d2 + a] / ratio
            val h = raw[3 * d2 + a] / ratio
            val left = (cx - w / 2f).coerceIn(0f, srcW.toFloat())
            val top = (cy - h / 2f).coerceIn(0f, srcH.toFloat())
            val right = (cx + w / 2f).coerceIn(0f, srcW.toFloat())
            val bottom = (cy + h / 2f).coerceIn(0f, srcH.toFloat())
            if (right <= left || bottom <= top) continue

            val label = CocoLabels.NAMES.getOrElse(bestCls) { "cls$bestCls" }
            candidates.add(Detection(RectF(left, top, right, bottom), bestCls, label, bestScore))
        }
        return nms(candidates, iouThreshold)
    }

    /** Class-agnostic greedy NMS (matches the reference nms behaviour). */
    private fun nms(dets: List<Detection>, iouThreshold: Float): List<Detection> {
        if (dets.isEmpty()) return emptyList()
        val sorted = dets.sortedByDescending { it.score }
        val picked = ArrayList<Detection>()
        val removed = BooleanArray(sorted.size)
        for (i in sorted.indices) {
            if (removed[i]) continue
            val a = sorted[i]
            picked.add(a)
            for (j in i + 1 until sorted.size) {
                if (removed[j]) continue
                if (iou(a.box, sorted[j].box) > iouThreshold) removed[j] = true
            }
        }
        return picked
    }

    private fun iou(a: RectF, b: RectF): Float {
        val x0 = max(a.left, b.left)
        val y0 = max(a.top, b.top)
        val x1 = min(a.right, b.right)
        val y1 = min(a.bottom, b.bottom)
        val inter = max(0f, x1 - x0) * max(0f, y1 - y0)
        val areaA = (a.right - a.left) * (a.bottom - a.top)
        val areaB = (b.right - b.left) * (b.bottom - b.top)
        return inter / (areaA + areaB - inter + 1e-6f)
    }

    @Synchronized
    override fun close() {
        if (handle != 0L) {
            nativeRelease(handle)
            handle = 0
            loadedModelPath = null
        }
    }

    private external fun nativeInit(): Long
    private external fun nativeLoad(handle: Long, modelPath: String, threads: Int)
    private external fun nativeRun(handle: Long, input: FloatArray, inH: Int, inW: Int): FloatArray?
    private external fun nativeRelease(handle: Long)

    companion object {
        const val INPUT_SIZE = 640
        // ultralytics letterbox padding color (gray 114 on all channels).
        private const val PAD_COLOR = 0xFF727272.toInt()

        init {
            System.loadLibrary("MNN")
            System.loadLibrary("yolo_jni")
        }
    }
}
